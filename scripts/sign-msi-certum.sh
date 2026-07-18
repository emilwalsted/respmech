#!/usr/bin/env bash
#
# Locally Authenticode-sign the RespMech Windows MSI with a Certum individual / Open Source
# code-signing certificate, then replace the unsigned MSI attached to its GitHub release.
#
# WHY LOCAL (not in CI): Certum individual/OSS keys are hardware-backed — a physical
# cryptoCertum card, or the SimplySign cloud HSM — with NO exportable .pfx (2023 CA/Browser
# Forum rule) and an interactive login. An ephemeral GitHub-hosted runner can't hold the card
# or complete the SimplySign TOTP session, so `release.yml` builds the MSI UNSIGNED and this
# script signs it on your Mac and re-uploads it. (Azure's individual signing path is
# US/Canada-only, so it isn't an option from Denmark.)
#
# ONE-TIME SETUP (macOS):
#   brew install jsign osslsigncode gh
#   gh auth login                         # if not already logged in
#   Physical card SKU:  plug in the cryptoCertum card + reader (macOS PC/SC finds it).
#   Cloud (SimplySign): brew install --cask simplysign ; open SimplySign Desktop ;
#                       log in with your userID + the mobile-app TOTP (a ~2 h session).
#
# USAGE:
#   scripts/sign-msi-certum.sh v2.1.1
#
#   Card (default):   CERTUM_PIN=1234 CERTUM_ALIAS="Open Source Developer, Emil ..." \
#                       scripts/sign-msi-certum.sh v2.1.1
#   Cloud (SimplySign): CERTUM_STORETYPE=PKCS11 CERTUM_PKCS11_CFG=~/certum-pkcs11.cfg \
#                       CERTUM_PIN=... CERTUM_ALIAS=... scripts/sign-msi-certum.sh v2.1.1
#
# NOTE: this script signs the MSI attached to a GitHub RELEASE (it downloads the asset and
# re-uploads the signed one), so it needs a tag. To sign a LOCAL msi — e.g. an artifact from
# a `workflow_dispatch` build with no tag — call jsign directly with the same flags. With the
# SimplySign session logged in, neither --alias nor --storepass is needed:
#
#   jsign --storetype PKCS11 --keystore ~/certum-pkcs11.cfg \
#         --tsaurl http://time.certum.pl/ --alg SHA-256 RespMech-<version>.msi
#   osslsigncode verify RespMech-<version>.msi     # must show a "Timestamp time:" line
#
# --tsaurl is NOT optional in practice: without it the signature carries no RFC-3161
# timestamp and dies when the certificate expires (osslsigncode then says
# "Timestamp is not available"). Add --replace to re-sign an already-signed file.
#
# ENV (all optional except where noted):
#   REPO               GitHub repo           (default: emilwalsted/respmech)
#   CERTUM_STORETYPE   CRYPTOCERTUM | PKCS11 (default: CRYPTOCERTUM — the physical card)
#   CERTUM_ALIAS       key/cert alias on the card/token. USUALLY UNNECESSARY: with a single
#                      key on the token jsign just uses it. (jsign is NOT a listing tool —
#                      running it without --alias does not print the aliases, it goes ahead
#                      and SIGNS. To list them: keytool -list -keystore NONE -storetype
#                      PKCS11 -providerClass sun.security.pkcs11.SunPKCS11 -providerArg <cfg>)
#   CERTUM_PIN         card / SimplySign PIN. Also usually unnecessary for PKCS11: while the
#                      SimplySign Desktop session is logged in, the token is already unlocked
#                      and jsign signs without prompting.
#   CERTUM_PKCS11_CFG  PKCS11 config file (PKCS11 storetype only; points at the SimplySign
#                      cryptoki .dylib — see the SimplySign Desktop install)
#   CERTUM_TSA         RFC-3161 timestamp URL (default: http://time.certum.pl/)
#
set -euo pipefail

TAG="${1:-}"
if [ -z "$TAG" ]; then
  echo "usage: scripts/sign-msi-certum.sh <release-tag>   (e.g. v2.1.1)" >&2
  exit 2
fi

REPO="${REPO:-emilwalsted/respmech}"
STORETYPE="${CERTUM_STORETYPE:-CRYPTOCERTUM}"
TSA="${CERTUM_TSA:-http://time.certum.pl/}"

for tool in jsign gh; do
  command -v "$tool" >/dev/null 2>&1 || { echo "error: '$tool' not found — brew install $tool" >&2; exit 1; }
done

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "==> Downloading the unsigned MSI from release $TAG …"
gh release download "$TAG" -R "$REPO" --pattern '*.msi' --dir "$WORK" --clobber
MSI="$(ls "$WORK"/*.msi 2>/dev/null | head -n1 || true)"
[ -n "$MSI" ] || { echo "error: no .msi asset found on release $TAG" >&2; exit 1; }
echo "    $(basename "$MSI")"

echo "==> Signing with Certum (storetype: $STORETYPE) …"
# jsign signs the file IN PLACE.
JSIGN_ARGS=(--storetype "$STORETYPE" --tsaurl "$TSA" --alg SHA-256)
[ -n "${CERTUM_ALIAS:-}" ] && JSIGN_ARGS+=(--alias "$CERTUM_ALIAS")
[ -n "${CERTUM_PIN:-}" ]   && JSIGN_ARGS+=(--storepass "$CERTUM_PIN")
if [ "$STORETYPE" = "PKCS11" ]; then
  [ -n "${CERTUM_PKCS11_CFG:-}" ] || { echo "error: CERTUM_PKCS11_CFG is required for PKCS11 (cloud) signing" >&2; exit 1; }
  JSIGN_ARGS+=(--keystore "$CERTUM_PKCS11_CFG")
fi
jsign "${JSIGN_ARGS[@]}" "$MSI"

echo "==> Verifying the Authenticode signature …"
if command -v osslsigncode >/dev/null 2>&1; then
  osslsigncode verify "$MSI"
else
  echo "    (osslsigncode not installed — skipping verify; brew install osslsigncode to enable)"
fi

echo "==> Replacing the release asset with the signed MSI …"
gh release upload "$TAG" -R "$REPO" "$MSI" --clobber

# Reflect the now-signed MSI in the release notes. release.yml writes the notes at build time,
# when the MSI is still UNSIGNED, so the Windows row carries no signing label (unlike the dmg,
# which is signed in CI). Now that the MSI is Certum-signed, flip that row to match reality —
# surgically (only the MSI token) and idempotently (a re-run is a no-op), and never fatally:
# if the row can't be found the notes are left untouched with a warning.
echo "==> Marking the MSI as signed in the release notes …"
NOTES_BODY="$(gh release view "$TAG" -R "$REPO" --json body --jq .body 2>/dev/null | tr -d '\r' || true)"
if [ -z "$NOTES_BODY" ]; then
  echo "::warning::no release notes found for $TAG; nothing to update."
elif printf '%s\n' "$NOTES_BODY" | grep -qE 'RespMech-\*\.msi`[[:space:]]*\(signed'; then
  echo "    notes already mark the MSI as signed — nothing to do."
elif printf '%s\n' "$NOTES_BODY" | grep -qF '`RespMech-*.msi`'; then
  printf '%s\n' "$NOTES_BODY" \
    | sed 's/`RespMech-\*\.msi`/`RespMech-*.msi` (signed + timestamped)/' > "$WORK/notes.md"
  gh release edit "$TAG" -R "$REPO" --notes-file "$WORK/notes.md"
  echo "    updated the Windows MSI row to '(signed + timestamped)'."
else
  echo "::warning::couldn't find the MSI row in the notes for $TAG; left notes unchanged."
fi

cat <<EOF

✅ Done — release $TAG now carries a Certum-signed MSI.

Note: a brand-new Certum OSS certificate has no Microsoft SmartScreen reputation yet, so
Windows may still show "unrecognised app" for the first downloads. Reputation builds up as
the signed installer is downloaded/run over time; the signature and publisher are valid.
EOF
