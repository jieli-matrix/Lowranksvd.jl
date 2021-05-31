module Lowranksvd
# Low Rank Singular Value Decomposition
using LinearAlgebra
import Base.show
"""
    LowRankSVD <: Factorization
Matrix factorization type of the singular value decomposition (SVD) approximation of a low rank matrix `A`.
LowRankSVD is return type of [`svd_lowrank(_)`](@ref), the corresponding matrix factorization function.
If `F::SVD` is the factorization object, `U`, `S`, `V` and `Vt` can be obtained
via `F.U`, `F.S`, `F.V` and `F.Vt`, such that `A ≈ U * Diagonal(S) * Vt`.
The singular values in `S` are sorted in descending order.
"""
struct LowRankSVD{T,Tr,M<:AbstractArray{T}} <: Factorization{T}
    U::M
    S::Vector{Tr}
    Vt::M
    function LowRankSVD{T,Tr,M}(U, S, Vt) where {T,Tr,M<:AbstractArray{T}}
      size(U,2) == size(Vt,1) == length(S) || throw(DimensionMismatch("$(size(U)), $(length(S)), $(size(Vt)) not compatible"))
      new{T,Tr,M}(U, S, Vt)
    end
end
LowRankSVD(U::AbstractArray{T}, S::Vector{Tr}, Vt::AbstractArray{T}) where {T,Tr} = LowRankSVD{T,Tr,typeof(U)}(U, S, Vt)
function LowRankSVD{T}(U::AbstractArray, S::AbstractVector{Tr}, Vt::AbstractArray) where {T,Tr}
    LowRankSVD(convert(AbstractArray{T}, U),
                convert(Vector{Tr}, S),
                convert(AbstractArray{T}, Vt))
end
function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, F::LowRankSVD{<:Any,<:Any,<:AbstractArray})
    summary(io, F); println(io)
    println(io, "U factor:")
    show(io, mime, F.U)
    println(io, "\nsingular values:")
    show(io, mime, F.S)
    println(io, "\nVt factor:")
    show(io, mime, F.Vt)
end
"""
    get_approximate_basis(A, l::Int64; niter::Int64 = 2, M = nothing) -> Q

Return Matrix ``Q``` with ``l`` orthonormal columns such that ``Q Q^H A`` approximates ``A``. If ``M`` is specified, then ``Q`` is such that ``Q Q^H (A - M)`` approximates ``A - M``.
    
# Arguments
- `A::AbstractArray{T}`: the input matrix of size ``(m, n)``.
- `l::Int64`: the dimension of subspace spanned by Q columns.
- `niter::Int64`(optional): the number of subspace iterations to conduct; `niter` must be a nonnegative integer. In most cases, the default value 2 is more than enough.
- `M::AbstractArray{T}`(optional): the input matrix of size ``(m, n)``.

# Examples
```jloctest
julia> A = rand(3,3);Q = get_approximate_basis(A,2)
3×2 Matrix{Float64}:
 -0.737784  -0.199989
 -0.451893  -0.563196
 -0.501465   0.801757
```

# References
- Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp, Finding structure with randomness: probabilistic algorithms for constructing approximate matrix decompositions, arXiv:0909.4061 [math.NA; math.PR], 2009 (available at `arXiv <http://arxiv.org/abs/0909.4061>`_).
"""
function get_approximate_basis(
    A::AbstractArray{T}, l::Int, niter::Int = 2, M::Union{AbstractArray{T}, Nothing} = nothing) where T
    m, n = size(A)
    Ω = rand(T, (n, l))

    Y = zeros(T, (m, l))
    Y_H = zeros(T, (n, l))

    if M === nothing 
        F_j = qr!(mul!(Y, A, Ω))
        for j = 1:niter
            F_H_j = qr!(mul!(Y_H, A', Matrix(F_j.Q)))
            F_j = qr!(mul!(Y, A, Matrix(F_H_j.Q)))
        end
    else
        Z = zeros(T, (m, l))
        Z_H = zeros(T, (n, l))
        F_j = qr!(mul!(Y, A, Ω) - mul!(Z, M, Ω))
        for j = 1:niter
            F_H_j = qr!(mul!(Y_H, A', Matrix(F_j.Q)) - mul!(Z_H, M', Matrix(F_j.Q)))
            F_j = qr!(mul!(Y, A, Matrix(F_H_j.Q)) - mul!(Z, M, Matrix(F_H_j.Q)))
        end
    end
    Matrix(F_j.Q)
end

function _low_rank_svd(A::AbstractArray{T}, l::Int64, niter::Int64 = 2, M::Union{AbstractArray{T}, Nothing} = nothing) where T
    
        m, n = size(A)
    if M === nothing
        Mt = nothing
    else
        Mt = transpose(M)
    end
    At = transpose(A)

    if m < n || n > l
        """
        computing the SVD approximation of a transpose in
        order to keep B shape minimal (the m < n case) or the V
        shape small (the n > l case)
        """
        Q = get_approximate_basis(At, l, niter, Mt)
        Qc = conj(Q)
        if M === nothing
            Bt = A * Qc
        else
            Bt = (A - M) * Qc
        end
        U, S, Vt = svd!(Bt)
        Vt = Vt * Q'
    else
        Q = get_approximate_basis(A, l, niter, M)
        if M === nothing
            B = Q' * A
        else
            B = Q' * (A - M)
        end

        U, S, Vt = svd!(B)
        U = Q * U
    end
    return LowRankSVD(U, S, Vt)
end

"""
    low_rank_svd(A, l::Int64; niter::Int64 = 2, M = nothing) -> LowRankSVD

Return the singular value decomposition LowRankSVD(U, S, Vt) of a matrix or a sparse matrix ``A`` such that ``A ≈ U diag(S) Vt``. In case ``M``` is given, then SVD is computed for the matrix ``A - M``.

# Arguments
- `A::AbstractArray{T}`: the input matrix of size ``(m, n)``.
- `l::Int64`: a slightly overestimated rank of A.
- `niter::Int64`(optional): the number of subspace iterations to conduct; niter must be a nonnegative integer, and defaults to 2.
- `M::AbstractArray{T}`(optional): the input matrix of size ``(m, n)``.

# Examples
```jloctest
julia> A = rand(4,3);F = low_rank_svd(A,2)
LowRankSVD{Float64, Float64, Matrix{Float64}}
U factor:
4×2 Matrix{Float64}:
 -0.225084   0.436754
 -0.278581  -0.846338
 -0.863012   0.0342116
 -0.356287   0.302964
singular values:
2-element Vector{Float64}:
 1.520187308842153
 0.26236654020151895
Vt factor:
2×3 Matrix{Float64}:
 -0.671183  -0.470594  -0.57276
  0.419535  -0.878149   0.229882
```

# References
- Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp, Finding structure with randomness: probabilistic algorithms for constructing approximate matrix decompositions, arXiv:0909.4061 [math.NA; math.PR], 2009 (available at `arXiv <http://arxiv.org/abs/0909.4061>`_).
"""
function low_rank_svd(A::AbstractArray{T}, l::Int64, niter::Int64 = 2, M::Union{AbstractArray{T}, Nothing} = nothing) where T
    return _low_rank_svd(A, l, niter, M)
end

end
