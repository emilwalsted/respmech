# Homebrew formula for the RespMech CLI.
#
# This lives in a tap: emilwalsted/homebrew-respmech (file: Formula/respmech.rb),
# so users install with:
#
#     brew install emilwalsted/respmech/respmech
#
# The formula installs the CLI into an isolated virtualenv under libexec. The
# heavy scientific dependencies (numpy/scipy/pandas) install as binary wheels, so
# no compiler is needed. The GUI is intentionally NOT bundled here (PySide6/Qt is
# large); GUI users install it with:  pip install "respmech[gui]".
#
# RELEASE NOTE: the `url`/`sha256` below point to the PyPI sdist and must be
# updated per release. The `resource` blocks for the Python dependencies are
# generated automatically after publishing to PyPI:
#
#     brew update-python-resources Formula/respmech.rb
#   (or: pip install homebrew-pypi-poet && poet respmech)
#
# The single resource stanza kept here is illustrative; run the command above to
# fill in the full, pinned dependency set with correct sha256 sums.

class Respmech < Formula
  include Language::Python::Virtualenv

  desc "Respiratory mechanics, work of breathing and diaphragm EMG entropy analysis"
  homepage "https://github.com/emilwalsted/respmech"
  url "https://files.pythonhosted.org/packages/source/r/respmech/respmech-2.0.0.tar.gz"
  sha256 "REPLACE_WITH_PYPI_SDIST_SHA256"
  license "GPL-3.0-or-later"

  depends_on "python@3.12"

  # >>> `brew update-python-resources` fills these in (numpy, scipy, pandas,
  #     openpyxl, tomli-w, et al.) after the PyPI release. <<<
  # resource "numpy" do
  #   url "https://files.pythonhosted.org/.../numpy-*.tar.gz"
  #   sha256 "..."
  # end

  def install
    virtualenv_install_with_resources
  end

  test do
    assert_match "respmech", shell_output("#{bin}/respmech --version")
    # migrate a tiny legacy settings file to TOML without executing it
    (testpath/"legacy.py").write <<~PY
      settings = {"input": {"format": {"samplingfrequency": 2000},
        "data": {"column_poes": 7, "column_pgas": 8, "column_pdi": 10,
                 "column_flow": 13, "column_volume": 14}}}
    PY
    system bin/"respmech", "migrate", testpath/"legacy.py", "-o", testpath/"out.toml"
    assert_predicate testpath/"out.toml", :exist?
  end
end
