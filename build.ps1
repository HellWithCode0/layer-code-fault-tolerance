$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    & python build_bibliography.py
    if ($LASTEXITCODE -ne 0) { throw "Bibliography build failed" }
    foreach ($name in @('supplement', 'supplement', 'supplement', 'manuscript', 'manuscript', 'manuscript')) {
        & pdflatex -interaction=nonstopmode -halt-on-error "$name.tex"
        if ($LASTEXITCODE -ne 0) { throw "PDF build failed: $name" }
    }
    & python finalize_pdf_links.py
    if ($LASTEXITCODE -ne 0) { throw "Link finalization failed" }
} finally { Pop-Location }
