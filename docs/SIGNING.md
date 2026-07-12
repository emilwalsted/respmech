# Code-signing the installers

The release workflow (`.github/workflows/release.yml`) signs the installers **only when the
relevant secrets are set**. With no secrets it builds an unsigned MSI + an ad-hoc-signed dmg
(the current behaviour). Add the secrets below and the next `v*` tag produces signed builds —
no workflow edit needed.

Add every secret under **GitHub ▸ repo ▸ Settings ▸ Secrets and variables ▸ Actions ▸ New
repository secret**. Nothing here is handled by the workflow author — you create the
certificates/accounts and paste the secrets yourself.

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

## Windows — Azure Artifact Signing (formerly "Trusted Signing")

~US$10/month (Basic tier, 5 000 signatures/mo), no hardware token — the key lives in
Microsoft's cloud HSM. Certificates chain to Microsoft's root, so SmartScreen reputation is
good from the start.

> **Eligibility:** originally organisations with 3+ years' verifiable history; individual
> identity validation is now available but confirm your status before subscribing. If it
> doesn't fit, **Certum Open Source Code Signing** (individual, ~€90–120/yr, physical token)
> is the fallback — that needs a different, token-based workflow step, not this one.

### One-time setup (Azure portal)

1. Create an **Artifact Signing** account (search "Artifact Signing"; formerly "Trusted
   Signing"). Pick the region — its endpoint host is `https://<code>.codesigning.azure.net/`
   (e.g. `weu` = West Europe, `eus` = East US). **The endpoint region must match the account.**
2. Complete **identity validation** → choose **Public Trust** (for public distribution).
3. Create a **certificate profile** under the account.
4. Create an **App registration** (Entra ID) → this is the service principal. Create a
   **client secret** on it (Certificates & secrets).
5. Grant that service principal the RBAC role **"Artifact Signing Certificate Profile Signer"**
   (older label: "Trusted Signing Certificate Profile Signer"), scoped to the account/profile.

### Secrets

| Secret | Value |
|---|---|
| `AZURE_TENANT_ID` | Entra tenant ID |
| `AZURE_CLIENT_ID` | the app registration's client (application) ID |
| `AZURE_CLIENT_SECRET` | the client secret from step 4 |
| `TRUSTED_SIGNING_ENDPOINT` | e.g. `https://weu.codesigning.azure.net/` |
| `TRUSTED_SIGNING_ACCOUNT` | the Artifact Signing account name |
| `TRUSTED_SIGNING_PROFILE` | the certificate profile name |

The workflow builds the MSI unsigned (briefcase can't drive cloud signing), then signs it in
place with `azure/artifact-signing-action@v2` and verifies the result with
`Get-AuthenticodeSignature`. Timestamping is mandatory (the cloud cert rotates ~every 3 days)
and is already configured in the workflow.

> This uses the client-secret path (simplest to set up). To avoid a stored secret you can
> switch to OIDC: add a federated credential on the app registration for this repo, add
> `permissions: id-token: write` to the `windows-msi` job, and an `azure/login@v3` step — then
> drop `AZURE_CLIENT_SECRET`.

---

## Turning it off / partial setup

The two platforms are independent: set only the Apple secrets and you get a signed dmg + an
unsigned MSI, or vice versa. Remove the secrets and the build reverts to unsigned/ad-hoc. The
GitHub release notes are generated to state, per platform, whether that artifact was signed.
