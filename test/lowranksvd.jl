using Test, LinearAlgebra, Random, Lowranksvd
using Lowranksvd: get_approximate_basis, _low_rank_svd, low_rank_svd
Random.seed!(1234)

function generate_low_rank_matrix(n::Int, m::Int, r::Int, eltya)
    ϵ = 5*eps(real(eltya))
    U = rand(eltya,(n,r))
    V = rand(eltya,(m,r))
    A = U*V' + rand(eltya, (n,m))*ϵ
    return A
end

@testset "lowrank svd" begin
    for eltya in (Float32, ComplexF32)
        # set tolerance
        rtol = 5*eps(real(eltya))
        approx_rtol = 100*rtol
        
        for (n, m, r) in ((64,32,5), (32,32,5))
            A = generate_low_rank_matrix(n, m, r, eltya)
            @testset "Q basis" begin
                Q = get_approximate_basis(A, r)
                #@test Q'*Q ≈ I(r) atol = approx_rtol
                @test norm(A - Q*Q'*A) < approx_rtol*norm(A)
            end
            
            @testset "svd test" begin
                F = low_rank_svd(A, r)
                A_approx = F.U * Diagonal(F.S) * F.Vt
                @test norm(A - A_approx) < approx_rtol*norm(A)
            end
            
        end
    end
end
