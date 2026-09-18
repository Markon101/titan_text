#![allow(dead_code)]
#[path="/data/data/com.termux/files/home/projects/titan_text/src/config.rs"] mod config;
#[path="/data/data/com.termux/files/home/projects/titan_text/src/field.rs"] mod field;
#[path="/data/data/com.termux/files/home/projects/titan_text/src/nca.rs"] mod nca;
#[path="/data/data/com.termux/files/home/projects/titan_text/src/vocab.rs"] mod vocab;
#[path="/data/data/com.termux/files/home/projects/titan_text/src/dataset.rs"] mod dataset;
#[path="/data/data/com.termux/files/home/projects/titan_text/src/checkpoint.rs"] mod checkpoint;

use anyhow::Result;
use candle_core::{Device,DType,Tensor};
use candle_nn::{VarMap,VarBuilder};
use dataset::SequenceDataset;
use field::MorphogenicField;
use nca::NeuralCellularAutomaton;
use vocab::TokenInterface;
use checkpoint::CheckpointManager;
fn logits(nca:&NeuralCellularAutomaton, interface:&TokenInterface, ids:&Tensor, steps:usize, dev:&Device)->Result<Tensor>{
 let f=MorphogenicField::from_tensor(interface.embed_tokens(ids)?,&nca.field_cfg);
 Ok(interface.logits(&nca.develop(&f,steps,dev)?.x)?)
}
fn main()->Result<()> {
 let dev=Device::Cpu;
 for name in ["v0_text","v0_viscous"] {
  let dir=format!("/data/data/com.termux/files/home/projects/titan_text/checkpoints/{name}");
  let m=CheckpointManager::load_manifest(&dir)?;
  let task=if m.task.is_empty(){"text"}else{m.task.as_str()};
  let ds=SequenceDataset::new(task);
  let mut vm=VarMap::new();
  let vb=VarBuilder::from_varmap(&vm,DType::F32,&dev);
  let nca=NeuralCellularAutomaton::new(vb.pp("nca"),&m.config.nca,&m.config.field)?;
  let ti=TokenInterface::new(vb.pp("interface"),ds.vocab.size(),m.config.field.channels)?;
  CheckpointManager::load_weights(&dir,&mut vm,&dev)?;
  let (ins,tgts)=ds.sample_val_batch(8,m.config.field.seq_len,&dev)?;
  let iv=ins.to_vec2::<u32>()?;let tv=tgts.to_vec2::<u32>()?;
  let l=m.config.field.seq_len;
  let mut matches=0;let mut interior=0;
  for b in 0..8 {for i in 0..l {if iv[b][(i+1)%l]==tv[b][i]{matches+=1;if i+1<l{interior+=1;}}}}
  let full_logits=logits(&nca,&ti,&ins,m.config.train.dev_steps,&dev)?;
  let model_acc=ti.accuracy(&full_logits,&tgts)?;
  let old_ids=ins.narrow(0,0,1)?;
  let mut changed=vec![iv[0].clone()];
  let i=l/2-1;
  let original_future=changed[0][i+1];
  let new_future=ds.vocab.encode("z")[0] as u32;
  changed[0][i+1]=if new_future!=original_future{new_future}else{ds.vocab.encode("q")[0] as u32};
  let new_ids=Tensor::new(changed.clone(),&dev)?;
  let old=logits(&nca,&ti,&old_ids,m.config.train.dev_steps,&dev)?;
  let new=logits(&nca,&ti,&new_ids,m.config.train.dev_steps,&dev)?;
  let old_cell=old.narrow(1,i,1)?;let new_cell=new.narrow(1,i,1)?;
  let max_diff=(&new_cell-&old_cell)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
  let rms_diff=(&new_cell-&old_cell)?.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt();
  let pred_old=old_cell.argmax(2)?.flatten_all()?.to_vec1::<u32>()?[0];
  let pred_new=new_cell.argmax(2)?.flatten_all()?.to_vec1::<u32>()?[0];
  println!("{}",serde_json::json!({"checkpoint":name,"step":m.cumulative_step,"val_model_acc":model_acc,"right_shift_copy_acc":matches as f32/(8*l) as f32,"right_shift_interior_acc":interior as f32/(8*(l-1)) as f32,"prediction_position":i,"changed_future_position":i+1,"future_old":ds.vocab.decode(&[original_future as usize]),"future_new":ds.vocab.decode(&[changed[0][i+1] as usize]),"prediction_old":ds.vocab.decode(&[pred_old as usize]),"prediction_new":ds.vocab.decode(&[pred_new as usize]),"same_prefix_logit_max_abs_change":max_diff,"same_prefix_logit_rms_change":rms_diff}));
 }

 {
  let dir="/data/data/com.termux/files/home/projects/titan_text/checkpoints/v0_viscous";
  let m=CheckpointManager::load_manifest(dir)?;
  let ds=SequenceDataset::new("text");
  let mut vm=VarMap::new();let vb=VarBuilder::from_varmap(&vm,DType::F32,&dev);
  let nca=NeuralCellularAutomaton::new(vb.pp("nca"),&m.config.nca,&m.config.field)?;
  let ti=TokenInterface::new(vb.pp("interface"),ds.vocab.size(),m.config.field.channels)?;
  CheckpointManager::load_weights(dir,&mut vm,&dev)?;
  let (ins,tgts)=ds.sample_val_batch(8,m.config.field.seq_len,&dev)?;

  {
   let lg=logits(&nca,&ti,&ins,m.config.train.dev_steps,&dev)?;
   let loss=ti.cross_entropy_loss(&lg,&tgts)?;
   let grads=loss.backward()?;
   let named=vm.data().lock().unwrap();
   let mut names=named.keys().cloned().collect::<Vec<_>>();names.sort();
   let mut entries=Vec::new();let mut gs=0f32;
   for name in names {
    let v=&named[&name];
    let g=grads.get(v).ok_or_else(||anyhow::anyhow!("unreachable variable {}",name))?;
    let gn=g.sqr()?.sum_all()?.to_scalar::<f32>()?;gs+=gn;
    let base=Tensor::from_vec(v.flatten_all()?.to_vec1::<f32>()?,v.shape(),&dev)?;
    entries.push((name,v.clone(),base,g.clone()));
   }
   drop(named);
   let norm=gs.sqrt();
   for eps in [0.01f64,0.003,0.001] {
    let mut losses=Vec::new();
    for sign in [1.0f64,-1.0] {
     for (_,v,base,g) in &entries {v.set(&(base+&(g*(sign*eps/norm as f64))?)?)?;}
     losses.push(ti.cross_entropy_loss(&logits(&nca,&ti,&ins,m.config.train.dev_steps,&dev)?,&tgts)?.to_scalar::<f32>()?);
    }
    for (_,v,base,_) in &entries {v.set(base)?;}
    let fd=(losses[0]-losses[1]) as f64/(2.*eps);
    println!("{}",serde_json::json!({"gradient_test":"CE directional derivative along global normalized gradient","checkpoint":"v0_viscous","steps":m.config.train.dev_steps,"viscosity":m.config.nca.viscosity,"reachable_tensors":entries.len(),"epsilon":eps,"autograd_derivative":norm,"finite_difference_derivative":fd,"relative_error":(fd-norm as f64).abs()/norm as f64}));
   }
   for (name,_,_,g) in &entries {println!("{}",serde_json::json!({"gradient_tensor":name,"l2":g.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt()}));}
  }
  for nu in [0.0f32,0.01,0.05,0.1,0.2,0.4]{
   let edited=nca.with_viscosity(nu);
   let lg=logits(&edited,&ti,&ins,m.config.train.dev_steps,&dev)?;
   println!("{}",serde_json::json!({"frozen_viscosity":nu,"val_acc":ti.accuracy(&lg,&tgts)?,"val_loss":ti.cross_entropy_loss(&lg,&tgts)?.to_scalar::<f32>()?}));
  }
  let (seed,_)=ds.sample_train_batch(1,m.config.field.seq_len,&dev)?;
  let mut f=MorphogenicField::from_tensor(ti.embed_tokens(&seed)?,&nca.field_cfg);
  for age in 0..=263 {
   if age==8 || age==263 {
    let e=f.energy()?;
    let dc=0.5*f.x.mean_keepdim(1)?.sqr()?.mean_all()?.to_scalar::<f32>()?;
    println!("{}",serde_json::json!({"checkpoint":"v0_viscous","development_age":age,"total_energy":e,"dc_energy":dc,"centered_energy":e-dc,"dc_energy_fraction":dc/e,"global_mean":f.x.mean_all()?.to_scalar::<f32>()?,"mean_max_amplitude":f.x.mean_keepdim(1)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?}));
   }
   if age<263 {f=nca.step_field(&f,&dev)?;f.x=f.x.detach();}
  }
 }
 for task in ["text","dyck"]{
  let ds=SequenceDataset::new(task);
  for val in [false,true]{
   let (x,y)=if val{ds.sample_val_batch(8,32,&dev)?}else{ds.sample_train_batch(8,32,&dev)?};
   let x=x.to_vec2::<u32>()?;let y=y.to_vec2::<u32>()?;
   let mut correct=0;for b in 0..8{for i in 0..32{if x[b][(i+1)%32]==y[b][i]{correct+=1;}}}
   println!("{}",serde_json::json!({"task":task,"split":if val{"val"}else{"train"},"right_shift_copy_acc":correct as f32/256.}));
  }
 }
 Ok(())
}
