module MCConfig

    export Config, FastUpdateConfig!, UpdateSlater!, GetSiteTuple, GetFockState, GetOccupancy

    using TightBindingToolkit, LinearAlgebra, Logging


    function GetFockState(occupancy::BitVector, localDim::Int64, sites::Int64)::Vector{Int64}

        range_starts   =   localDim .* (collect(0:sites-1)) .+ 1
        range_ends     =   localDim .* (collect(1:sites))
        ##### Resolving a localDim * sites Bitvector into a vector of size sites, where each element is a BitVector of size localDim.
        resolvedOccupancy  =   getindex.(Ref(occupancy), UnitRange.(range_starts,  range_ends))
        ##### Getting the fock state index from a Bitvector of size localDim on each site which represents which orbitals are occupied.
        FockState   =   Int.(getindex.( getproperty.(resolvedOccupancy , :chunks) , 1))

        return FockState
    end


    function GetOccupancy(fockState::Vector{Int64}, localDim::Int64)::BitVector
        ##### Converts a vector of ints representing the fock state on each site to a vector of bits representing which orbitals are occupied.
        return   BitVector(vcat(digits.(fockState, base = 2, pad = localDim)...))
    end


    mutable struct Config{}

        ParticleCount   ::  Int64   ##### Number of particles filling the system
        Occupancy       ::  BitVector   ##### A vector of bits of size localDim * sites, where each bit represents whether the orbital is occupied or not.
        FockState       ::  Vector{Int64}   ##### A vector of size sites, where each element is an integer representing the fock state of the site.
        Kel             ::  Dict{Int64, Int64}  ##### A dictionary mapping the filled particle index to the site * local index.
        SlaterMat       ::  Matrix{ComplexF64}  ##### The Slater determinant matrix.
        W               ::  Matrix{ComplexF64}  ##### The W matrix relevant for VMC updates.

        function Config(n::Int64, Occupancy::BitVector, lattice::Lattice{T}, Hamiltonian::LatticeHamiltonian; states::Vector{Int64} = collect(1:n)) where {T}
            @assert sum(Occupancy) == n "Occupied orbitals must be equal to the number of particles"
            @assert length(Occupancy) == length(Hamiltonian.bands) "Total number of orbitals must be equal to the Hamiltonian dimensions."
            ##### Getting the fock state index from a Bitvector of size localDim on each site which represents which orbitals are occupied.
            FockState   =   GetFockState(Occupancy, lattice.uc.localDim, lattice.length)
            ##### The order of fermions being filled into the vacuum state.
            occupied    =   findall(==(1), Occupancy)
            Kel         =   Dict{Int64, Int64}()
            setindex!.(Ref(Kel), occupied, 1:n)
            ##### The Slater determinant is the determinant of the SlaterMat.
            SlaterMat   =   getindex(Hamiltonian.states, occupied, states)
            @assert !isapprox(abs(det(SlaterMat)), 0.0, atol = 1e-6, rtol = 1e-6) "The Slater determinant must be non-zero when initializing a configuration."
            ##### Matrix to keep track of during Monte Carlo runs
            W           =   Hamiltonian.states[:, states] * inv(SlaterMat)

            return new{}(n, Occupancy, FockState, Kel, SlaterMat, W)

        end

    end


    function GetSiteTuple(index::Int64, localDim::Int64)
        ##### Returns the site tuple (r,o) from the site index, where r is the physical site and o is the orbital.
        return ((((index .- 1) .÷ localDim) .+ 1), ((index .- 1) .% localDim) .+ 1)
    end


    ##### This function updates the configuration of the system after a move is accepted.
    function FastUpdateConfig!(config::Config, particles::Vector{Int64}, NewPositions::Vector{Int64}, lattice::Lattice{T}) where {T}
        ##### get the current positions of the particles being moved
        currentPositions = getindex.(Ref(config.Kel), particles)
        ##### change the occupancy of the orbitals, and move the particles to their new positions.
        config.Occupancy[currentPositions] .= Ref(0)
        config.Occupancy[NewPositions] .= Ref(1)
        ##### update the Kel dictionary
        setindex!.(Ref(config.Kel), NewPositions, particles)
        ##### update the Fock state
        config.FockState    =  GetFockState(config.Occupancy, lattice.uc.localDim, lattice.length)

        C       =   config.W[NewPositions, particles]
        b       =   -inv(C) * ( config.W[NewPositions,:] - Matrix(1.0I , size(config.W) )[particles,:] )
        config.W=   config.W + config.W[: , particles] * b

    end


    function UpdateSlater!(config::Config, Hamiltonian::LatticeHamiltonian; states::Vector{Int64} = collect(1:n))
        ##### The updated Slater determinant matrix keeping into account the potential reordering of particles.
        config.Slater   =   getindex(Hamiltonian.states, getindex.(Ref(config.Kel), 1:config.ParticleCount), states)
        ##### The updated W matrix keeping into account the potential reordering of particles.
        config.W        =   Hamiltonian.states[:, states] * inv(config.Slater)

    end

















































end