use crate::config::TitanConfig;
use anyhow::{Context, Result};
use candle_core::Device;
use candle_nn::VarMap;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::process::Command;

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelManifest {
    pub schema_version: u32,
    pub git_commit: String,
    pub random_seed: u64,
    pub cumulative_step: usize,
    pub dev_steps: usize,
    pub train_loss: f32,
    pub train_accuracy: f32,
    pub val_loss: f32,
    pub val_accuracy: f32,
    pub grad_norm: f32,
    pub state_energy: f32,
    pub param_count: usize,
    pub checkpoint_hash: String,
    pub timestamp_unix: u64,
    pub config: TitanConfig,
    #[serde(default)]
    pub task: String,
}

pub struct CheckpointManager;

impl CheckpointManager {
    /// Queries current git commit hash
    pub fn get_git_commit() -> String {
        Command::new("git")
            .args(["rev-parse", "--short", "HEAD"])
            .output()
            .ok()
            .and_then(|out| String::from_utf8(out.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "uncommitted".to_string())
    }

    /// Computes simple hash string of saved safetensors file
    pub fn compute_file_hash(path: &str) -> String {
        if let Ok(bytes) = fs::read(path) {
            let mut hasher: u64 = 0xcbf29ce484222325;
            for b in bytes {
                hasher ^= b as u64;
                hasher = hasher.wrapping_mul(0x100000001b3);
            }
            format!("{:016x}", hasher)
        } else {
            "unknown".to_string()
        }
    }

    /// Saves the model weights (SafeTensors) and manifest (JSON) atomically
    pub fn save(
        dir: &str,
        manifest_partial: &mut ModelManifest,
        varmap: &VarMap,
    ) -> Result<()> {
        fs::create_dir_all(dir).with_context(|| format!("Failed to create dir '{}'", dir))?;

        let model_path = format!("{}/model.safetensors", dir);
        let manifest_path = format!("{}/manifest.json", dir);
        let tmp_manifest_path = format!("{}/manifest.json.tmp", dir);

        // 1. Save safetensors weights
        varmap.save(&model_path).with_context(|| format!("Failed to save safetensors to '{}'", model_path))?;

        // 2. Compute checkpoint hash
        manifest_partial.checkpoint_hash = Self::compute_file_hash(&model_path);

        // 3. Save manifest atomically via tmp file
        let json_str = serde_json::to_string_pretty(manifest_partial)?;
        fs::write(&tmp_manifest_path, json_str)?;
        fs::rename(&tmp_manifest_path, &manifest_path)?;

        Ok(())
    }

    /// Loads the manifest from a checkpoint directory
    pub fn load_manifest(dir: &str) -> Result<ModelManifest> {
        let manifest_path = format!("{}/manifest.json", dir);
        let json_str = fs::read_to_string(&manifest_path)
            .with_context(|| format!("Failed to read manifest at '{}'", manifest_path))?;
        let manifest: ModelManifest = serde_json::from_str(&json_str)
            .with_context(|| "Failed to parse checkpoint manifest JSON")?;

        if manifest.schema_version != SCHEMA_VERSION {
            anyhow::bail!(
                "Unsupported schema_version {} (expected {})",
                manifest.schema_version,
                SCHEMA_VERSION
            );
        }

        Ok(manifest)
    }

    /// Loads safetensors weights into a VarMap
    pub fn load_weights(dir: &str, varmap: &mut VarMap, _device: &Device) -> Result<()> {
        let model_path = format!("{}/model.safetensors", dir);
        if !Path::new(&model_path).exists() {
            anyhow::bail!("Model file '{}' not found", model_path);
        }
        varmap.load(&model_path).with_context(|| format!("Failed to load safetensors from '{}'", model_path))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manifest_backward_compatibility() -> Result<()> {
        let json = r#"{
            "schema_version": 1,
            "git_commit": "abc1234",
            "random_seed": 42,
            "cumulative_step": 50,
            "dev_steps": 8,
            "train_loss": 0.05,
            "train_accuracy": 0.95,
            "val_loss": 0.1,
            "val_accuracy": 0.9,
            "grad_norm": 0.02,
            "state_energy": 2.5,
            "param_count": 43000,
            "checkpoint_hash": "hash123",
            "timestamp_unix": 1789000000,
            "config": {
                "field": { "seq_len": 32, "channels": 64, "periodic_boundary": true },
                "nca": { "hidden_dim": 96, "step_size": 0.5, "update_rate": 1.0, "activation": "gelu", "viscosity": 0.01 },
                "train": { "epochs": 50, "dev_steps": 8, "batch_size": 8, "lr": 0.003, "weight_decay": 0.01, "save_every": 50, "log_every": 10, "seed": 42 },
                "probe": { "horizon": 64, "perturbation_eps": 0.01, "lp_window": 8, "forcing_amplitude": 0.2 }
            }
        }"#;

        let manifest: ModelManifest = serde_json::from_str(json)?;
        assert_eq!(manifest.cumulative_step, 50);
        assert_eq!(manifest.config.nca.viscosity, 0.01);
        assert_eq!(manifest.task, "");
        Ok(())
    }

    #[test]
    fn test_save_and_load_checkpoint_roundtrip() -> Result<()> {
        let dev = Device::Cpu;
        let config = TitanConfig::default();
        let varmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &dev);
        let _interface = crate::vocab::TokenInterface::new(vb.pp("test_iface"), 10, config.field.channels)?;

        let temp_path = std::env::temp_dir().join(format!("titan_test_roundtrip_{}_{}", std::process::id(), rand::random::<u32>()));
        let temp_dir = temp_path.to_str().unwrap();
        let mut manifest = ModelManifest {
            schema_version: SCHEMA_VERSION,
            git_commit: "test_git".to_string(),
            random_seed: 42,
            cumulative_step: 25,
            dev_steps: 8,
            train_loss: 0.1,
            train_accuracy: 0.9,
            val_loss: 0.2,
            val_accuracy: 0.85,
            grad_norm: 0.05,
            state_energy: 1.5,
            param_count: 100,
            checkpoint_hash: "".to_string(),
            timestamp_unix: 123456,
            config: config.clone(),
            task: "text".to_string(),
        };

        CheckpointManager::save(temp_dir, &mut manifest, &varmap)?;
        let loaded_manifest = CheckpointManager::load_manifest(temp_dir)?;
        assert_eq!(loaded_manifest.cumulative_step, 25);
        assert_eq!(loaded_manifest.task, "text");

        let mut load_varmap = VarMap::new();
        let vb_load = candle_nn::VarBuilder::from_varmap(&load_varmap, candle_core::DType::F32, &dev);
        let _load_iface = crate::vocab::TokenInterface::new(vb_load.pp("test_iface"), 10, config.field.channels)?;
        CheckpointManager::load_weights(temp_dir, &mut load_varmap, &dev)?;

        let _ = std::fs::remove_dir_all(temp_dir);
        Ok(())
    }
}
