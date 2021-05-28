using Lowranksvd
using Documenter

DocMeta.setdocmeta!(Lowranksvd, :DocTestSetup, :(using Lowranksvd); recursive=true)

makedocs(;
    modules=[Lowranksvd],
    authors="jieli-matrix <li_j20@fudan.edu.cn> and contributors",
    repo="https://github.com/jieli-matrix/Lowranksvd.jl/blob/{commit}{path}#{line}",
    sitename="Lowranksvd.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://jieli-matrix.github.io/Lowranksvd.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jieli-matrix/Lowranksvd.jl",
)
