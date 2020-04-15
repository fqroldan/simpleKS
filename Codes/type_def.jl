using QuantEcon

mutable struct Pars
	β::Float64
	γ::Float64

	α::Float64
	δ::Float64

	ρz::Float64
	σz::Float64
	ρϵ::Float64
	σϵ::Float64
end

mutable struct Grids
	a::Vector{Float64}	# Idiosyncratic endogenous state
	ϵ::Vector{Float64}	# Idiosyncratic exogenous state
	k::Vector{Float64}	# Aggregate endogenous state
	z::Vector{Float64}	# Aggregate exogenous state
	N::Vector{Int64}	
	n::Dict{Symbol, Int64}
	Pϵ::Matrix{Float64}
	Pz::Matrix{Float64}
end

mutable struct KS
	pars::Pars

	gr::Grids

	k′::Array{Float64, 2}
	r::Array{Float64, 2}
	w::Array{Float64, 2}

	vf::Array{Float64, 4}
	Nϕ::Int64
	ϕ::Vector{Array{Float64, 4}}
	n::Dict{Symbol, Int64}
end
function move_grids!(xgrid; xmin=0.0, xmax=1.0)
	xgrid[:] = xgrid[:] * (xmax-xmin) .+ xmin
 	nothing
 end

function KS(;
	β::Float64 = 0.96,
	γ::Float64 = 2.0,
	α::Float64 = 0.33,
	δ::Float64 = 0.05,
	ρz = 0.9,
	σz = 0.025,
	Na = 7,
	Nϵ = 5,
	Nk = 10,
	Nz = 5)
	
	ρϵ = 0.9136		# Floden-Lindé for US
	σϵ = 0.0426		# Floden-Lindé for US
	pars = Pars(β, γ, α, δ, ρz, σz, ρϵ, σϵ)

	amin, amax = -0.0, 3.0

	agrid = cdf.(Beta(2,1), range(0,1,length=Na))
	move_grids!(agrid, xmax=amax, xmin=amin)
	kgrid = range(0.1, 2.0, length=Nk)

	ϵ_chain = tauchen(Nϵ, ρϵ, σϵ, 0, 2)
	z_chain = tauchen(Nz, ρz, σz, 0, 2)
	
	Pϵ = ϵ_chain.p
	ϵgrid = exp.(ϵ_chain.state_values)
	Pz = z_chain.p
	zgrid = exp.(z_chain.state_values)

	N = [Na, Nϵ, Nk, Nz]
	grnames = Dict(:a => 1, :ϵ => 2, :k => 3, :z => 4)

	gr = Grids(agrid, ϵgrid, kgrid, zgrid, N, grnames, Pϵ, Pz)

	k′ = ones(Nk, Nz)
	r, w = [ones(Nk, Nz) for jj in 1:2]

	vf = zeros(Na, Nϵ, Nk, Nz)

	Nϕ = 2
	ϕ = [zeros(size(vf)) for jj in 1:Nϕ]
	polnames = Dict(:a => 1, :c => 2)

	return KS(pars, gr, k′, r, w, vf, Nϕ, ϕ, polnames)
end

# Utilities to manipulate ϕ's
get_ϕ(ks::KS, sym::Symbol) = ks.ϕ[ks.n[sym]]
function set_ϕ!(ks, sym::Symbol, y::Array)
	if size(y) == size(ks.ϕ[ks.n[sym]])
		ks.ϕ[ks.n[sym]] = y
	else
		throw(error("Wrong size"))
	end
	nothing
end
