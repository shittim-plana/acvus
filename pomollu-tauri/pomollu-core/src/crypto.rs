//! AES-256-GCM encryption for the on-disk token store.
//!
//! Format: `nonce (12 bytes) || ciphertext+tag`. Each encryption draws a
//! fresh random nonce — there is no legacy zero-nonce fallback (Pomollu has
//! no pre-existing token stores to migrate).
//!
//! The key is derived as SHA-256 of a fixed in-app password. This is
//! obfuscation, not a hardware-backed secret: the real isolation boundary is
//! the Android app-private data directory. Keystore-backed keys would be the
//! next enforcement level (residual).

use aes_gcm::{
    aead::{Aead, KeyInit},
    Aes256Gcm, Nonce,
};
use rand::RngCore;
use sha2::{Digest, Sha256};

use crate::error::PomolluError;

const NONCE_LEN: usize = 12;

fn derive_key(password: &str) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(password.as_bytes());
    hasher.finalize().into()
}

pub fn encrypt(data: &[u8], password: &str) -> Result<Vec<u8>, PomolluError> {
    let key = derive_key(password);
    let cipher = Aes256Gcm::new_from_slice(&key).map_err(|_| PomolluError::AesEncrypt)?;
    let mut nonce_bytes = [0u8; NONCE_LEN];
    rand::rng().fill_bytes(&mut nonce_bytes);
    let nonce = Nonce::from_slice(&nonce_bytes);
    let encrypted = cipher
        .encrypt(nonce, data)
        .map_err(|_| PomolluError::AesEncrypt)?;
    let mut result = nonce_bytes.to_vec();
    result.extend(encrypted);
    Ok(result)
}

pub fn decrypt(data: &[u8], password: &str) -> Result<Vec<u8>, PomolluError> {
    if data.len() < NONCE_LEN {
        return Err(PomolluError::AesDecrypt);
    }
    let key = derive_key(password);
    let cipher = Aes256Gcm::new_from_slice(&key).map_err(|_| PomolluError::AesDecrypt)?;
    let (nonce_bytes, ciphertext) = data.split_at(NONCE_LEN);
    let nonce = Nonce::from_slice(nonce_bytes);
    cipher
        .decrypt(nonce, ciphertext)
        .map_err(|_| PomolluError::AesDecrypt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let plaintext = b"hello pomollu";
        let encrypted = encrypt(plaintext, "pw").unwrap();
        assert!(encrypted.len() >= plaintext.len() + NONCE_LEN + 16);
        assert_eq!(decrypt(&encrypted, "pw").unwrap(), plaintext);
    }

    #[test]
    fn unique_nonces() {
        let a = encrypt(b"same", "k").unwrap();
        let b = encrypt(b"same", "k").unwrap();
        assert_ne!(&a[..NONCE_LEN], &b[..NONCE_LEN]);
    }

    #[test]
    fn wrong_password_fails() {
        let encrypted = encrypt(b"secret", "correct").unwrap();
        assert!(decrypt(&encrypted, "wrong").is_err());
    }

    #[test]
    fn short_data_fails() {
        assert!(decrypt(&[0u8; 11], "pw").is_err());
        assert!(decrypt(&[], "pw").is_err());
    }
}
