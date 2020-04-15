using QuantEcon, Random, Interpolations, Distributions, BasisMatrices, DataFrames, GLM, Optim, PlotlyJS, ColorSchemes, Printf

include("type_def.jl")

function value(ks::KS, ap, yv, jϵ, jz, kpv, itp_v)
	pars, gr = ks.pars, ks.gr
	β, γ = pars.β, pars.γ

	c = budget_constraint(yv, ap)

	if c <= 0
		return -1e10 + c
	else
		Ev = 0.0
		for (jϵp, ϵpv) in enumerate(gr.ϵ), (jzp, zpv) in enumerate(gr.z)
			Ev += gr.Pϵ[jϵ, jϵp] * gr.Pz[jz, jzp] * itp_v(ap, ϵpv, kpv, zpv)
		end

		v = c^(1-γ) / (1-γ) + β * Ev
		return v
	end
end

budget_constraint(yv, ap) = yv-ap
CoH(av, ϵv, rv, wv) = (1+rv)*av + ϵv*wv

function opt_value(ks::KS, itp_v)
	gr = ks.gr

	Jgrid = gridmake(1:size(gr.k,1), 1:size(gr.z,1))

	new_v = similar(ks.vf)
	new_a = similar(ks.ϕ[ks.n[:a]])
	new_c = similar(ks.ϕ[ks.n[:c]])

	amin = minimum(gr.a)
	amax = maximum(gr.a)

	for js in 1:size(Jgrid,1)
		jk, jz = Jgrid[js, :]
		kv = gr.k[jk]
		zv = gr.z[jz]
		rv = ks.r[jk, jz]
		wv = ks.w[jk, jz]

		wv, rv = ks.w[jk, jz], ks.r[jk, jz]
		kp = ks.k′[jk, jz]

		for (ja, av) in enumerate(gr.a), (jϵ, ϵv) in enumerate(gr.ϵ)

			yv = CoH(av, ϵv, rv, wv)

			if amin > min(yv, amax)
				println(amax, amin, yv)
			end
			amax = min(yv, amax)

			res = Optim.optimize(
				ap -> -value(ks, ap, yv, jϵ, jz, kp, itp_v),
				amin, amax, GoldenSection()
				)

			ap = res.minimizer
			cc = budget_constraint(yv, ap)

			new_v[ja, jϵ, jk, jz] = -res.minimum
			new_a[ja, jϵ, jk, jz] = ap

			new_c[ja, jϵ, jk, jz] = cc
		end
	end

	set_ϕ!(ks, :a, new_a)
	set_ϕ!(ks, :c, new_c)

	return new_v
end

function vfi_iter(ks::KS)
	gr = ks.gr
	knots = (gr.a, gr.ϵ, gr.k, gr.z)

	itp_v = interpolate(knots, ks.vf, Gridded(Linear()))

	new_v = opt_value(ks, itp_v)

	return new_v
end

function update_vϕ!(ks::KS, new_v; upd_η=0.9)
	ks.vf = ks.vf + upd_η * (new_v - ks.vf)
end

function vfi!(ks::KS; maxiter = 2500, tol::Float64=1e-4)
	dist = 1+tol
	iter = 0
	upd_η = 0.75

	while dist > tol && iter < maxiter
		iter += 1

		old_v = copy(ks.vf)
		new_v = vfi_iter(ks)

		norm_v = sqrt( sum(old_v.^2) )
		dist = sqrt( sum((new_v-old_v).^2) ) / norm_v

		update_vϕ!(ks, new_v, upd_η = upd_η)
	end
	return dist
end

function iter_simul(ks::KS, λ, k0, z0, itp_a, aϵ_grid, basis, Qϵ)
	ga = [itp_a(aϵ_grid[jaϵ,1], aϵ_grid[jaϵ,2], k0, z0) for jaϵ in 1:size(aϵ_grid,1)]

	k1 = ga'λ
	k1 = max(min(k1, maximum(ks.gr.k)), minimum(ks.gr.k))

	ϵ1 = rand(Normal(0,1))

	z1 = exp( ks.pars.ρz * log(z0) + ks.pars.σz * ϵ1 )
	z1 = max(min(z1, maximum(ks.gr.z)), minimum(ks.gr.z))

	savings = max.(min.(ga, maximum(ks.gr.a)), minimum(ks.gr.a))

	Qa = BasisMatrix(basis, Expanded(), savings, 0).vals[1]
	Q = row_kron(Qϵ, Qa)

	λ1 = Q' * λ

	return λ1, k1, z1
end

function simul(ks::KS)
	gr = ks.gr
	Random.seed!(1)

	burn_in = 100
	simul_length = 1000

	Na = 1000

	agrid_fine = range(minimum(gr.a), maximum(gr.a), length=Na)
	Qϵ = kron(gr.Pϵ, ones(Na,1))

	λ = ones(Na*gr.N[2])
	λ = λ / sum(λ)

	aϵ_grid = gridmake(agrid_fine, gr.ϵ)

	k0 = mean(gr.k)
	z0 = mean(gr.z)

	knots = (gr.a, gr.ϵ, gr.k, gr.z)
	itp_a = interpolate(knots, ks.ϕ[ks.n[:a]], Gridded(Linear()))
	
	basis = Basis(LinParams(agrid_fine, 0))
	for tt in 1:burn_in
		λ, k0, z0 = iter_simul(ks, λ, k0, z0, itp_a, aϵ_grid, basis, Qϵ)
	end

	k_vec, z_vec = [Vector{Float64}(undef, simul_length) for jj in 1:2]

	for tt in 1:simul_length
		k_vec[tt] = k0
		z_vec[tt] = z0
		λ, k0, z0 = iter_simul(ks, λ, k0, z0, itp_a, aϵ_grid, basis, Qϵ)
	end

	return k_vec, z_vec
end

function new_LoM_k(ks::KS, β::Vector)
	gr = ks.gr
	new_k = similar(ks.k′)

	for (jk, kv) in enumerate(gr.k), (jz, zv) in enumerate(gr.z)
		Khat = exp( β'log.([kv, zv]) )
		Khat = max(min(Khat, maximum(gr.k)), minimum(gr.k))
		new_k[jk, jz] = Khat
	end
	return new_k
end

function update_LoM_k!(ks::KS, new_k; upd_η = 0.5)

	norm = sqrt( sum(ks.k′.^2) )
	dist = sqrt( sum((new_k-ks.k′).^2) ) / norm

	ks.k′ = ks.k′ + upd_η * (new_k - ks.k′)
	return dist
end


function update_k!(ks::KS)
	# Simulate and get time series for (K_t, z_t)
	k_vec, z_vec = simul(ks)

	# Run regressions
    df = DataFrame(k=log.(k_vec), k_lag = [0; log.(k_vec[1:end-1])], z = log.(z_vec))
	ols = lm(@formula(k ~ -1 + k_lag + z), df)

	# Figure out new LoM
	new_k = new_LoM_k(ks, coef(ols))

	# Update ks.k′
	dist = update_LoM_k!(ks, new_k)
	return dist
end

function update_prices!(ks::KS; upd_η = 0.75)
	gr = ks.gr
	new_w = similar(ks.w)
	new_r = similar(ks.r)

	α = ks.pars.α

	Lv = 1

	Jgrid = gridmake(1:size(gr.k,1), 1:size(gr.z,1))

	for js in 1:size(Jgrid, 1)
		jk, jz = Jgrid[js, :]
		kv = gr.k[jk]
		zv = gr.z[jz]

		F_L = (1-α) * zv * (kv/Lv)^α
		F_K =   α   * zv * (kv/Lv)^(α-1)

		new_w[jk, jz] = F_L
		new_r[jk, jz] = F_K - ks.pars.δ
	end

	dist_w = sum((new_w - ks.w).^2) / sum((ks.w).^2)
	dist_r = sum((new_r - ks.r).^2) / sum((ks.r).^2)

	dist = max(dist_w, dist_r)

	ks.w = ks.w + upd_η * (new_w - ks.w)
	ks.r = ks.r + upd_η * (new_r - ks.r)

	return dist
end

function eqm!(ks::KS; maxiter::Int64=250, tol::Float64=1e-4)

	dist = 1+tol
	iter = 0

	tol_vfi = 5e-2
	dist_v, dist_K = zeros(2)


	while dist > tol && iter < maxiter
		iter += 1

		# Not needed for now but would be if labor supply endogenous
		dist_p = update_prices!(ks)

		dist_v = vfi!(ks, tol = tol_vfi)
		dist = max(dist_v, dist_p)

		dist_K = update_k!(ks)

		dist = max(dist, dist_K)

		println("Iteration $(iter): d(v,k) = $([@sprintf("%.3g", dv) for dv in [dist_v, dist_K]])")

		tol_vfi = max(0.9*tol_vfi, 1e-6)
	end
end
