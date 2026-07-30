# Code-signing the installers

The two platforms sign very differently:

- **macOS — signed in CI, secret-gated.** With the Apple secrets set (below), the release
  workflow (`.github/workflows/release.yml`) produces a Developer ID-signed + notarised dmg;
  with no secrets it falls back to an ad-hoc dmg. Add the secrets under **GitHub ▸ repo ▸
  Settings ▸ Secrets and variables ▸ Actions** and the next `v*` tag signs automatically.
- **Windows — signed LOCALLY after release.** CI always builds an *unsigned* MSI, because the
  chosen certificate (Certum individual/OSS) is hardware-backed and can't sign on a CI runner
  (see below). You run `scripts/sign-msi-certum.sh <tag>` on your Mac to sign it and replace
  the release asset. No GitHub secrets are involved on the Windows side.

You create the certificates/accounts and hold the keys yourself — nothing here is done for you.

---

## macOS — Developer ID signing + notarisation

Requires an **Apple Developer Program** membership ($99/yr). This replaces the ad-hoc dmg with
a signed + notarised one that opens with no Gatekeeper warning.

### One-time setup

1. In the Apple Developer portal (or Xcode ▸ Settings ▸ Accounts ▸ Manage Certificates),
   create a **Developer ID Application** certificate. Export it **with its private key** as a
   `.p12` (Keychain Access ▸ right-click the cert ▸ Export), set an export password.
2. Base64-encode the `.p12` for storage as a secret:
   ```bash
   base64 -i DeveloperID.p12 | pbcopy      # now in your clipboard
   ```
3. Create an **app-specific password** at <https://appleid.apple.com> ▸ Sign-In and Security ▸
   App-Specific Passwords (this is NOT your Apple ID password).
4. Note your 10-character **Team ID** (Apple Developer ▸ Membership).

### Secrets

| Secret | Value |
|---|---|
| `APPLE_DEV_ID_IDENTITY` | `Developer ID Application: Your Name (TEAMID1234)` — exact string, incl. the `(TEAMID)` |
| `APPLE_CERTIFICATE_P12_BASE64` | the base64 blob from step 2 |
| `APPLE_CERTIFICATE_PASSWORD` | the `.p12` export password |
| `APPLE_ID` | your Apple ID email |
| `APPLE_TEAM_ID` | the 10-char Team ID |
| `APPLE_APP_SPECIFIC_PASSWORD` | the app-specific password from step 3 |
| `KEYCHAIN_PASSWORD` | any random string (password for the throwaway CI keychain) |

The workflow imports the cert into a temporary keychain, signs the `.app` with a hardened
runtime (`briefcase package macOS --identity … --no-notarize`), then notarises + staples the
dmg itself with `xcrun notarytool submit --wait` + `xcrun stapler staple`.

> If signing fails with "unable to build certificate chain", also import Apple's **WWDR
> intermediate** certificate into the keychain step. Notarisation can take several minutes;
> the `--wait` blocks until Apple returns a verdict.

---

## Windows — Certum individual / Open Source Code Signing (signed locally)

**Why not signed in CI.** Azure's individual signing path is USA & Canada only (unavailable in
Denmark). The individual-friendly EU option is **Certum Open Source Code Signing** — but since
the June 2023 CA/Browser Forum rules, every code-signing key (EV *and* OSS) must live on a
FIPS hardware module with **no exportable `.pfx`**. A Certum key sits either on a physical
cryptoCertum card or in Certum's SimplySign cloud (reached via an interactive TOTP login).
Neither can sign on an ephemeral GitHub-hosted runner, so **CI builds the MSI unsigned and you
sign it locally on your Mac** with `scripts/sign-msi-certum.sh`. No GitHub secrets involved.

### One-time setup

1. **Buy the certificate** at <https://shop.certum.eu> → *Open Source Code Signing*. Two SKUs
   that matter (individuals only; a GitHub project URL is accepted as OSS proof):
   - **Set** (~€69) — physical cryptoCertum card + reader. **Recommended**: a plugged-in card
     is the simplest to sign with (just a PIN), no per-session cloud login, no monthly cap.
   - **Cloud** (~€49) — key in SimplySign cloud; no hardware, but each signing session needs a
     mobile-app TOTP login (~2 h session) and there's a 5 000-signature/month cap.
   Identity validation: government photo ID + proof of address + a short video/liveness check.
   (Validity term is changing in 2026 — confirm at checkout.)
2. **Install the tools** on your Mac:
   ```bash
   brew install jsign osslsigncode gh
   ```
   - **Card**: plug in the cryptoCertum card + reader (macOS PC/SC detects it).
   - **Cloud**: `brew install --cask simplysign`, open **SimplySign Desktop**, log in with your
     userID + the TOTP from the SimplySign mobile app, and note the PKCS#11 cryptoki `.dylib`
     path it installs (needed for a `PKCS11` config file).

### Signing a release

After a `v*` tag has published (the release carries the **unsigned** MSI), run:

```bash
# physical card (default)
scripts/sign-msi-certum.sh v2.1.1

# SimplySign cloud instead
CERTUM_STORETYPE=PKCS11 CERTUM_PKCS11_CFG=~/certum-pkcs11.cfg \
  scripts/sign-msi-certum.sh v2.1.1
```

Neither `CERTUM_ALIAS` nor `CERTUM_PIN` is normally needed. There is one key on the token, so
jsign uses it without being told which, and while the SimplySign Desktop session is logged in
the token is already unlocked. Set them only when something asks for them.

**If you do need the alias, `jsign` will not tell you.** Run without `--alias` and it does not
list anything, it goes ahead and signs. `keytool` is the listing tool:

```bash
keytool -list -keystore NONE -storetype PKCS11 \
  -addprovider SunPKCS11 -providerArg ~/certum-pkcs11.cfg
```

(On a JDK older than 9, `-providerClass sun.security.pkcs11.SunPKCS11 -providerArg <cfg>`
instead of `-addprovider`. For the physical card, the same command with the card's own PKCS#11
config.) It asks for a keystore password; with the SimplySign session logged in, an empty one
goes through. The alias is the text before the first comma of the entry line, and on SimplySign
it is the key's hex id rather than a readable name:

```
Keystore provider: SunPKCS11-SimplySign
Your keystore contains 1 entry
41E1C9A2E3A20873F608F834D4B06507, PrivateKeyEntry,
```

That hex id is the certificate's own serial number, so `osslsigncode verify` prints it back as
`Serial:` under the signer. One entry means jsign needs no `--alias` at all. Passing it anyway
costs nothing and keeps the command correct if a renewed certificate ever puts a second key on
the token.

The script downloads the MSI from the release, Authenticode-signs it with
`jsign` (RFC-3161 timestamp `http://time.certum.pl/`), verifies with `osslsigncode`, and
re-uploads it over the unsigned asset (`gh release upload --clobber`).

It refuses to start while `release.yml` is still building for that tag, and says why. That run
uploads the MSI it built with `--clobber`, so a signature applied in the meantime is replaced
by an unsigned asset minutes later, and nothing about the release looks wrong afterwards: same
name, same version, no signature. A second argument `--force` signs anyway. The case that made
this worth checking was v2.3.3, whose tag was moved onto a fix, which ran both workflows again.

Two things in a successful run look like problems and are not. `osslsigncode` opens with
`Warning: MsiDigitalSignatureEx stream doesn't exist`: that second, optional MSI signature
stream is one jsign does not write, and Windows validates the signature without it. And the
certificate is only valid for a year (`notAfter` a year after issue). The signature outlives it
anyway, which is the whole reason `--tsaurl` is not optional: a timestamped signature keeps
verifying after the certificate expires, so the only thing that needs renewing on time is the
ability to sign something new.

What a good run ends with, worth comparing against next time:

```
	Timestamp time: Jul 30 09:49:55 2026 GMT
Timestamp Server Signature verification: ok
Signature CRL verification: ok
Signature verification: ok
Number of verified signatures: 1
Succeeded
```

> **SmartScreen reputation.** A brand-new Certum OSS certificate has no Microsoft SmartScreen
> reputation, so the first downloads may still show "unrecognised app" even though the
> signature and publisher are valid — reputation accrues as the signed installer is used. This
> is why the release notes keep the *More info ▸ Run anyway* hint regardless.

---

## Summary

| Platform | Where | What you provide |
|---|---|---|
| **macOS** | CI, on a `v*` tag | The 7 `APPLE_*` / `KEYCHAIN_PASSWORD` secrets above → signed + notarised dmg. No secrets → ad-hoc dmg. |
| **Windows** | Locally, after release | A Certum OSS cert (card or cloud) + `scripts/sign-msi-certum.sh`. Until you sign, the released MSI is unsigned. |

The platforms are fully independent — you can ship a signed+notarised dmg while the MSI is
still unsigned, or vice versa.
