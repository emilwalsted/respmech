#!/usr/bin/env python3
"""Is the changelog entry exhaustive and written for a reader?

    python3 tools/check_changelog.py                  # Unreleased, against commits since the last tag
    python3 tools/check_changelog.py --version v2.4.0 # that dated entry, against its own range
    python3 tools/check_changelog.py --warnings-only   # never exit non-zero (for a soft CI gate)

Exit 0 = every user-visible change in the range is represented in the entry.
Exit 1 = something shipped that the entry does not mention.
Exit 2 = the entry, the range or the repository could not be read.

WHY
---
CHANGELOG.md is the single source for what a release says: docs/RELEASING.md folds
its entry into the dated section, and respmech.dk's changelog page (plus the
mailing-list notification built from it) is the reader-facing rewrite of the same
list. So a change that never reaches the entry is a change nobody is ever told
about. Nothing checked that, and "did I remember everything?" is exactly the
question a human answers worst at the end of a release.

HOW, AND WHAT IT CANNOT DO
--------------------------
This is a heuristic, and it says so rather than pretending otherwise. It walks
the commits in the range, decides which ones are user-visible from the paths
they touch, pulls distinctive words out of each one (its subject, and the public
names it adds), and asks whether the entry mentions any of them. Anything with
no trace at all is reported.

It can therefore be wrong in both directions: an entry that describes a change
in entirely different words will be flagged, and a bullet that merely name-drops
a symbol without explaining it will pass. It is a net for the thing that
actually happens — a commit quietly left out — not a judge of prose. A deliberate
omission is recorded, not silenced:

    <!-- changelog-skip <sha7> <reason, at least a few words> -->

placed anywhere in CHANGELOG.md. The reason is required, so the next person can
see why something was left out instead of guessing.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHANGELOG = os.path.join(ROOT, 'CHANGELOG.md')

# Paths whose changes a user of the app never sees. Everything else counts as
# user-visible, so a NEW top-level directory errs towards being checked rather
# than towards being ignored silently.
USYNLIGE = (
    'tests/', 'docs/', '.github/', 'tools/', '.gitignore', '.gitattributes',
    'CLAUDE.md', 'CHANGELOG.md', 'README.md', 'LICENSE', '.editorconfig',
    '.pre-commit-config.yaml', 'ruff.toml', '.vscode/',
)
# Words too common to be evidence that a bullet describes a given commit.
STOPORD = set("""
a an and are as at be been but by can could did do does for from had has have how i if in
into is it its just make makes made not now of on only or our out over per so than that the
their them then there these they this to too up use used uses using was were what when where
which while who why will with without would you your add adds added also again all any
after before both each even ever fix fixes fixed get gets got let lets like more most much
new no non none nor off one other same see set sets should since some still such take takes
that's very via when whether across against along already although always among another
because been being below between during else enough every few first further here however
instead least less many maybe mean means might must never next once other's rather really
right said say says second seem seems several she since so soon still sure than that's
themselves though three thus together toward two under until upon us usually way well went
whole whose within yet chore refactor cleanup tidy wip bump minor major patch release
""".split())

# Public names a changelog bullet would plausibly mention, from added lines.
NAVNE = [
    re.compile(r'^\+\s*def\s+([a-zA-Z_][\w]*)'),
    re.compile(r'^\+\s*class\s+([A-Za-z_][\w]*)'),
    re.compile(r'^\+\s*([a-z_][\w]*)\s*[:=]\s'),          # module-level setting/const
    re.compile(r'^\+.*["\']([a-z_][a-z0-9_]{4,})["\']\s*[:=]'),   # settings key
]


def kør(args, **kw):
    # git's own output (commit messages, diffs) is always UTF-8, regardless of the
    # process's locale. Without an explicit encoding, text=True decodes with
    # locale.getpreferredencoding(), which on Windows CI runners defaults to cp1252 —
    # not UTF-8. A commit containing any of the typographic characters this project's
    # history is full of (·, –, —, ›, →) then crashes with an uncaught
    # UnicodeDecodeError, which is a real, uncaught exit code 1 — even though this tool
    # is meant to be an informational, --warnings-only step that must never fail a
    # branch (measured: reproduced the exact crash by forcing a non-UTF-8 default
    # locale). Decode as UTF-8 unconditionally, with a lossy fallback so a genuinely
    # malformed byte still can't crash a check whose entire job is printing a warning.
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=120,
                          encoding='utf-8', errors='replace', **kw)


def fejl(besked, kode=2):
    print(f'FEJL: {besked}', file=sys.stderr)
    sys.exit(kode)


def tags():
    r = kør(['git', 'tag', '--sort=-v:refname', '--list', 'v*'])
    return [t for t in r.stdout.split() if t]


def spænd(version: str | None):
    """(fra, til, beskrivelse) for det commit-interval entrien skal dække."""
    t = tags()
    if version is None:
        if not t:
            return None, 'HEAD', 'hele historikken (der findes ingen tags)'
        return t[0], 'HEAD', f'{t[0]}..HEAD'
    v = 'v' + version.strip().lstrip('vV')
    if v not in t:
        # Den vigtigste gang, man vil koere denne kontrol, er lige FOER man tagger: entrien
        # er foldet til en dateret sektion, saa der er ingen "Unreleased" tilbage at maale,
        # og tagget findes endnu ikke. At fejle dér gjorde vaerktoejet ubrugeligt i netop det
        # oejeblik (opdaget under klargoeringen af v2.3.3). Maal derfor mod HEAD i stedet, og
        # sig at det er det, der sker.
        if not t:
            fejl('der findes ingen v*-tags at maale imod')
        print(f'note: {v} er ikke tagget endnu, saa intervallet maales til HEAD.\n')
        return t[0], 'HEAD', f'{t[0]}..HEAD (endnu ikke tagget som {v})'
    i = t.index(v)
    forrige = t[i + 1] if i + 1 < len(t) else None
    if forrige is None:
        return None, v, f'alt til og med {v}'
    return forrige, v, f'{forrige}..{v}'


def commits(fra, til):
    spec = f'{fra}..{til}' if fra else til
    # --no-merges: en merge-commits indhold ligger i dens forældre, så den ville
    # tælle hver ændring to gange og oversvømme rapporten med dubletter.
    r = kør(['git', 'log', '--no-merges', '--format=%H%x1f%s', spec])
    if r.returncode != 0:
        fejl(f'kunne ikke læse commits i {spec}: {" ".join(r.stderr.split())[:160]}')
    ud = []
    for linje in r.stdout.splitlines():
        if '\x1f' in linje:
            sha, emne = linje.split('\x1f', 1)
            ud.append((sha.strip(), emne.strip()))
    return ud


def stier(sha):
    r = kør(['git', 'show', '--name-only', '--format=', '-z', sha])
    return [s for s in r.stdout.split('\0') if s]


def brugersynlig(paths):
    """Rører commit'et noget, en bruger af appen kan mærke?"""
    rørte = [p for p in paths if not any(p == u or p.startswith(u) for u in USYNLIGE)]
    return bool(rørte), rørte


# De to steder versionsnummeret står. Begge hører til brugersynlig kode, og
# release-commit'et rører netop dem og intet andet.
VERSIONSFILER = {'pyproject.toml', 'src/respmech/__init__.py'}
VERSIONSLINJE = re.compile(r'^[+-]\s*(?:__version__|version)\s*=\s*["\'][^"\']+["\']\s*$')


def er_versionsbump(sha: str, rørte) -> bool:
    """Er commit'et selve versionsopdateringen, og altså ikke en ændring at beskrive?

    Release-commit'et hæver `__version__` og `[tool.briefcase] version`. Begge
    ligger i filer, der ellers er brugersynlige, så uden denne regel melder
    kontrollen release-commit'et som en ændring uden spor i entrien, og porten i
    publish-pypi.yml stopper udgivelsen. Målt, ikke gættet: v2.3.3 blev tagget, og
    PyPI fik ingenting, fordi kontrollen fældede sin egen release.

    Reglen er snæver med vilje. Rører commit'et andet end de to filer, er den ude af
    spil, og rører det KUN dem, skal hver enkelt ændret linje være en
    versionstildeling. En rigtig ændring i `__init__.py` slipper altså ikke igennem,
    blot fordi den ligger i samme commit som et versionsnummer."""
    if set(rørte) - VERSIONSFILER:
        return False
    d = kør(['git', 'show', '--format=', '--unified=0', sha,
             '--', *sorted(VERSIONSFILER)]).stdout
    linjer = [linje for linje in d.splitlines()
              if linje[:1] in '+-' and not linje.startswith(('+++', '---'))]
    return bool(linjer) and all(VERSIONSLINJE.match(linje) for linje in linjer)


def emne_ord(emne: str) -> set:
    """Ord fra commit-emnet. Forfatterens eget resumé er det mest pålidelige
    signal der findes; ord fra diffen er langt støjende (en variabel ved navn
    `channel` fik et commit om volumen-trend til at "matche" et punkt om
    EKG-kanaler, målt her 30-07-2026)."""
    ud = set()
    for w in re.findall(r'[A-Za-z_][A-Za-z0-9_]*', emne):
        lw = w.lower()
        if len(lw) > 2 and lw not in STOPORD:
            ud.add(lw)
    return ud


def symboler(sha: str) -> set:
    """Nye offentlige navne, commit'et tilføjer. Et enkelt af dem i et punkt er
    stærkt bevis (`ecg_auto_detect` står ikke tilfældigt i en sætning), hvor
    almindelige ord kræver flere sammenfald."""
    ud = set()
    d = kør(['git', 'show', '--format=', '--unified=0', sha]).stdout
    for linje in d.splitlines():
        if not linje.startswith('+') or linje.startswith('+++'):
            continue
        for m in NAVNE:
            g = m.match(linje)
            if g:
                n = g.group(1).lower()
                if len(n) > 5 and '_' in n and n not in STOPORD and not n.startswith('test'):
                    ud.add(n)
    return ud


def har_ord(linje: str, ord_: str) -> bool:
    """Ordgrænse-match, så "main" ikke rammer "domain"."""
    return re.search(r'\b' + re.escape(ord_) + r'\w*', linje) is not None


def punkter(tekst: str):
    """Entriens punkter, ét ad gangen, med fortsættelseslinjer samlet.

    Matchningen sker PR. PUNKT, ikke mod hele entrien. Mod hele entrien var
    kontrollen praktisk taget blind: da et punkt om advanced-dialogen blev
    slettet i en prøve, bestod commit'et alligevel, fordi ordet "checkbox" stod i
    et helt andet punkt om noget helt andet. Et commit er repræsenteret, når ÉN
    linje handler om det, ikke når dets ord er spredt ud over teksten."""
    ud, nu = [], []
    for linje in tekst.splitlines():
        s = linje.strip()
        if s.startswith(('-', '*')):
            if nu:
                ud.append(' '.join(nu))
            nu = [s.lstrip('-* ').strip()]
        elif s and nu:
            nu.append(s)
        elif not s and nu:
            ud.append(' '.join(nu))
            nu = []
    if nu:
        ud.append(' '.join(nu))
    return [p.lower() for p in ud if p]


def entry(version: str | None):
    """(overskrift, tekst) for entrien der skal kontrolleres."""
    if not os.path.isfile(CHANGELOG):
        fejl('CHANGELOG.md findes ikke')
    t = open(CHANGELOG, encoding='utf-8').read()
    if version is None:
        # Den SIDSTE "## Unreleased" er den levende; skabelonen i HTML-kommentaren
        # ovenfor bærer samme overskrift og må ikke forveksles med den.
        fund = list(re.finditer(r'^##\s+Unreleased\s*$(.*?)(?=^##\s|\Z)', t, re.S | re.M))
        if not fund:
            fejl('CHANGELOG.md har ingen "## Unreleased"-sektion')
        return 'Unreleased', fund[-1].group(1)
    v = 'v' + version.strip().lstrip('vV')
    m = re.search(r'^##\s+' + re.escape(v) + r'\b[^\n]*$(.*?)(?=^##\s|\Z)', t, re.S | re.M)
    if not m:
        fejl(f'CHANGELOG.md har ingen "## {v}"-sektion')
    return v, m.group(1)


def waivers():
    """{sha7: begrundelse} fra <!-- changelog-skip … --> i CHANGELOG.md."""
    if not os.path.isfile(CHANGELOG):
        return {}
    t = open(CHANGELOG, encoding='utf-8').read()
    ud = {}
    for m in re.finditer(r'<!--\s*changelog-skip\s+([0-9a-f]{7,40})\s+(.+?)\s*-->', t, re.S):
        grund = ' '.join(m.group(2).split())
        if len(grund.split()) >= 3:
            ud[m.group(1)[:7]] = grund
    return ud


def sidst_rørt_changelog():
    r = kør(['git', 'log', '-1', '--format=%H %cI', '--', 'CHANGELOG.md'])
    p = r.stdout.split()
    return (p[0], p[1]) if len(p) >= 2 else (None, None)


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--version', help='kontrollér en dateret entry i stedet for Unreleased')
    ap.add_argument('--warnings-only', action='store_true',
                    help='rapportér, men afslut altid 0')
    ap.add_argument('--strict', action='store_true',
                    help='lad også svagt dækkede ændringer være en fejl')
    a = ap.parse_args(argv)

    if kør(['git', 'rev-parse', '--git-dir']).returncode != 0:
        fejl('dette er ikke et git-arbejdstræ')

    # En overfladisk klon har ingen tags og næsten ingen historik, og så måler denne
    # kontrol ingenting. Målt i en `git clone --depth 1` af dette repo: uden
    # --version meldte den "1 commits, 1 til gennemsyn" og var grøn, altså et
    # fuldstændig indholdsløst grønt lys, og MED --version fejlede den med "er ikke
    # et tag i dette repo" — hvilket i en release-port ville betyde, at hver eneste
    # udgivelse blev stoppet af værktøjet selv. Derfor siges det højt her.
    # `actions/checkout` henter som standard dybde 1; workflowene sætter
    # fetch-depth: 0 af netop denne grund.
    if kør(['git', 'rev-parse', '--is-shallow-repository']).stdout.strip() == 'true':
        fejl('dette er en overfladisk (shallow) klon, så hverken tags eller historik er '
             'fuldt tilgængelige, og kontrollen ville måle ingenting.\n'
             '       Kør `git fetch --unshallow --tags`, eller sæt `fetch-depth: 0` på '
             'actions/checkout i workflowet.')
    if not tags():
        fejl('der findes ingen v*-tags i denne klon, så der er intet interval at måle '
             'imod.\n       Kør `git fetch --tags`, eller sæt `fetch-depth: 0` på '
             'actions/checkout i workflowet.')

    navn, tekst = entry(a.version)
    fra, til, beskrivelse = spænd(a.version)
    fritaget = waivers()

    liste = commits(fra, til)
    if not liste:
        print(f'Intet at kontrollere: {beskrivelse} indeholder ingen commits.')
        return 0

    linjer = punkter(tekst)

    # HVAD DENNE KONTROL KAN, OG HVAD DEN IKKE KAN.
    # En ordsammenligning kan pålideligt afgøre ÉN ting: at et commit slet ikke har
    # noget sprogligt overlap med entrien. Alt derudover er gæt. Undervejs prøvede
    # jeg tre skarpere regler, og alle tre kunne narres på det samme datasæt: to
    # fælles ord i et langt punkt, og senere "unikke" ord, som blev unikke netop
    # fordi det rigtige punkt var slettet. Så kontrollen fælder kun den dom, den kan
    # stå på mål for, og afleverer resten som et arbejdsark, sorteret så det
    # svageste match står først. Mennesket læser seks linjer; det er billigt, og
    # det er ærligt.
    tilsyn, mangler, sprunget, waived = [], [], [], []
    for sha, emne in liste:
        paths = stier(sha)
        synlig, rørte = brugersynlig(paths)
        if not synlig:
            sprunget.append((sha[:7], emne, ', '.join(paths[:3])))
            continue
        if er_versionsbump(sha, rørte):
            sprunget.append((sha[:7], emne, 'kun versionsnummeret'))
            continue
        if sha[:7] in fritaget:
            waived.append((sha[:7], emne, fritaget[sha[:7]]))
            continue

        ord_ = emne_ord(emne)
        sym = symboler(sha)
        bedst_k, bedst_traf, bedst_grund = None, [], ''
        for k, linje in enumerate(linjer):
            s = sorted(x for x in sym if har_ord(linje, x))
            o = sorted(x for x in ord_ if har_ord(linje, x))
            if s:
                # Et nyt offentligt navn i et punkt er det stærkeste bevis, der findes:
                # `ecg_auto_detect` står ikke tilfældigt i en sætning.
                bedst_k, bedst_traf, bedst_grund = k, s + o, 'nyt navn'
                break
            if len(o) > len(bedst_traf):
                bedst_k, bedst_traf, bedst_grund = k, o, 'emneord'
        if not bedst_traf:
            mangler.append((sha[:7], emne, (sorted(sym) + sorted(ord_))[:7], rørte[:3]))
        else:
            stærk = bedst_grund == 'nyt navn'
            tilsyn.append((0 if stærk else len(bedst_traf), sha[:7], emne, bedst_traf[:4],
                           bedst_grund, linjer[bedst_k][:64], rørte[:3], stærk))
    # Svageste først: de usikre match skal læses, de stærke skal blot bekræftes.
    tilsyn.sort(key=lambda r: (1 if r[7] else 0, r[0]))
    svage = [r for r in tilsyn if not r[7] and r[0] < 3]

    print(f'== Changelog-kontrol: "{navn}" mod {beskrivelse} ==\n')
    print(f'{len(liste)} commits: {len(tilsyn)} brugersynlige til gennemsyn '
          f'(heraf {len(svage)} med svagt match), {len(mangler)} uden spor, '
          f'{len(waived)} bevidst udeladt, {len(sprunget)} ikke brugersynlige.\n')

    if tilsyn:
        print('Gennemgang, svageste match først. Bekræft at punktet faktisk dækker ændringen:')
        for antal, sha, emne, traf, grund, linje, rørte, stærk in tilsyn:
            mærke = 'ok  ' if stærk else ('?   ' if antal < 3 else '~   ')
            print(f'  {mærke} {sha}  {emne[:64]}')
            print(f'         nærmeste punkt, via {grund} "{", ".join(traf)}":')
            print(f'           {linje}…')
    if waived:
        print('\nBevidst udeladt (changelog-skip):')
        for sha, emne, grund in waived:
            print(f'  ---  {sha}  {emne[:66]}')
            print(f'         fordi: {grund}')
    if sprunget:
        # Listet, ikke skjult: klassificeringen er et skøn, og et skøn der ikke kan
        # ses, kan ikke bestrides.
        print('\nIkke brugersynlige, sprunget over (rører kun tests, docs, CI eller værktøj):')
        for sha, emne, p in sprunget:
            print(f'  ·    {sha}  {emne[:58]}   [{p}]')
    if mangler:
        print('\nUDEN SPOR I ENTRIEN:')
        for sha, emne, tk, rørte in mangler:
            print(f'  MANGLER {sha}  {emne}')
            print(f'          rører: {", ".join(rørte)}')
            print(f'          entrien nævner intet af: {", ".join(tk) or "(ingen særkendetegn fundet)"}')

    # Er der landet brugersynlig kode EFTER at changelog'en sidst blev rørt, kan
    # entrien pr. definition være forældet, uanset hvad ordsammenligningen siger.
    cl_sha, cl_tid = sidst_rørt_changelog()
    if cl_tid and a.version is None:
        senere = []
        for sha, emne in liste:
            if sha[:7] == (cl_sha or '')[:7]:
                continue
            synlig, _ = brugersynlig(stier(sha))
            if not synlig:
                continue
            t = kør(['git', 'log', '-1', '--format=%cI', sha]).stdout.strip()
            if t and t > cl_tid:
                senere.append((sha[:7], emne))
        if senere:
            print(f'\nNote: {len(senere)} brugersynlige commit(s) landede EFTER at CHANGELOG.md '
                  f'sidst blev opdateret ({cl_tid[:16]}):')
            for sha, emne in senere[:8]:
                print(f'  ·    {sha}  {emne[:66]}')
            print('  Entrien kan altså være forældet, også hvor ordene tilfældigt passer.')

    haardt = list(mangler) + (list(svage) if a.strict else [])
    if haardt:
        if mangler:
            print(f'\nIKKE i orden: {len(mangler)} brugersynlige ændring(er) har INTET spor i '
                  f'"{navn}".\nSkriv dem ind, eller registrér en bevidst udeladelse med '
                  '<!-- changelog-skip <sha7> <begrundelse> --> i CHANGELOG.md.', file=sys.stderr)
        if a.strict and svage:
            print(f'--strict: {len(svage)} svagt dækkede ændring(er) tæller også som fejl.',
                  file=sys.stderr)
        return 0 if a.warnings_only else 1
    if svage:
        print(f'\nIntet uden spor, men {len(svage)} ændring(er) står til gennemsyn ovenfor.')
        return 0
    print(f'\nI orden: alt brugersynligt i {beskrivelse} er dækket af et punkt i "{navn}".')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
