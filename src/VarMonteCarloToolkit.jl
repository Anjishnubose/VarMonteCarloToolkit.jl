module VarMonteCarloToolkit

# Write your package code here.
    include("MCConfig.jl")
    using .MCConfig
    export Config, FastUpdateConfig!, UpdateSlater!, GetSiteTuple, GetFockState, GetOccupancy

    include("MCEstimator.jl")
    using .MCEstimator
    export LocalEstimator

end
