# GPG Signing Guide

All release tags and release commits must be GPG-signed. This document covers setup for maintainers and verification for consumers.

## Maintainer Setup

### 1. Generate a GPG key (if you don't have one)

```bash
gpg --full-generate-key
# Select: RSA and RSA, 4096 bits, no expiry (or set one and rotate)
# Use the email associated with your GitHub account
```

### 2. Configure Git to use your key

```bash
# List keys to find your key ID
gpg --list-secret-keys --keyid-format long

# Configure git
git config --global user.signingkey <KEY_ID>
git config --global commit.gpgsign true
git config --global tag.gpgSign true
```

### 3. Upload your public key to GitHub

```bash
# Export your public key
gpg --armor --export <KEY_ID>
```

Paste the output at: **GitHub > Settings > SSH and GPG keys > New GPG key**

### 4. (Optional) Publish to a keyserver

```bash
gpg --keyserver keys.openpgp.org --send-keys <KEY_ID>
```

## Signing a Release Tag

```bash
git tag -s v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0
```

The `-s` flag creates a signed tag. Git will prompt for your GPG passphrase.

## Verifying Tags and Commits

### Verify a tag

```bash
git tag -v v0.1.0
```

Expected output includes `Good signature from ...` and the signer identity.

### Verify a commit

```bash
git log --show-signature -1 <commit-hash>
```

### Verify with a specific trust level

If the signer's key is not in your keyring, import it first:

```bash
# From GitHub
curl https://github.com/<username>.gpg | gpg --import

# Then verify
git tag -v v0.1.0
```

## Key Rotation

When rotating keys:

1. Generate a new key.
2. Sign the new key with the old key: `gpg --sign-key <NEW_KEY_ID>`.
3. Update GitHub with the new public key.
4. Announce the rotation in the CHANGELOG for the next release.
5. Revoke the old key after a transition period: `gpg --gen-revoke <OLD_KEY_ID>`.

## Troubleshooting

- **"gpg: signing failed: No secret key"** -- Ensure `user.signingkey` matches a key in `gpg --list-secret-keys`.
- **"gpg: signing failed: Inappropriate ioctl for device"** -- Set `export GPG_TTY=$(tty)` in your shell profile.
- **macOS pinentry issues** -- Install `pinentry-mac`: `brew install pinentry-mac` and set it in `~/.gnupg/gpg-agent.conf`.
