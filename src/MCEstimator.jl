module MCEstimator

    using TightBindingToolkit, LinearAlgebra, Logging, SparseArrays

    using ..VarMonteCarloToolkit.MCConfig: Config, GetFockState, GetOccupancy


    ##### The estimator of the operator O is given by ∑_{x'} [<x'| O | x>*|x'|/|x|], where |x| is the the Slater determinant of a Fock state |x>.
    function LocalEstimator(config::Config, sites::Vector{Int64}, Operators::Vector{SparseMatrixCSC{Float64, Int64}})::ComplexF64
        ##### sites is a vector of lattice sites on which the operators act.
        ##### Operators is a vector of local operators acting on the sites. Each operator is a sparse matrix in the full local Fock basis.

        FinalFockSubStates = kron(selectdim.(adjoint.(Operators), 2, config.FockState[sites])...)    ##### the final Fock state in the subspace where the operators act after the action of the operators. The non-zero indices of this sparse vector tells us the final posisble Fock sub-states |x'> where the operators take the current configuration, |x> to, and the corresponding non-zero values tell us the expectation value <x'| O | x>.
        localFockDimensions     =   Operators[begin].m
        localDim    =   log2(localFockDimensions)

        estimator::ComplexF64   =   0.0 + 0.0im
        ##### TODO : This loop can be broadcasted?
        for finalSubState in FinalFockSubStates.nzind
            ##### The final fock state is the same in all the sites where the operators do not act.
            finalFullFockState  = copy(config.FockState)
            finalFullFockState[sites] = reverse(digits(finalSubState - 1, localFockDimensions, pad = length(sites)))    ##### a given final fock sub-state is converted into a vector of ints representing the final fock state on each site where the operators act.

            expectation = FinalFockSubStates[finalSubState]     ##### the expectation value <x'| O | x> for the final Fock sub-state |x'>.
            
            finalOccupancy  =   GetOccupancy(finalFullFockState, localDim)  ##### the final occupancy vector in the sites sub-space for the final Fock state |x'>.
            @assert sum(finalOccupancy) == config.ParticleCount "Operators being measured must conserve the number of particles."

            change  =   finalOccupancy - config.Occupancy    ##### the change in the occupancy vector in the sites sub-space.
            annihilated =   findall(==(-1), change)    ##### the orbitals that are annihilated by the operators.
            created =   findall(==(1), change)    ##### the orbitals that are created by the operators.

            movedParticles  =   getindex.(Ref(config.kel), annihilated)
            SlaterRatio     =   det(config.W[movedParticles, created])    ##### the ratio of the Slater determinant matrix elements for the final and initial Fock states.
            estimator       +=  expectation * SlaterRatio    ##### the expectation value of the operator O.
        end

        return estimator

    end
































































end