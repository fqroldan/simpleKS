using QuantEcon, Interpolations, Distributions, Optim, PlotlyJS, ColorSchemes

include("type_def.jl")

function value(ks::KS, ap, yv, jϵ, jz, kpv)
	pars, gr = ks.pars, ks.gr
	β, γ = pars.β, pars.γ

	c = budget_constraint(yv, ap)

	if c <= 0
		return -1e6 + c
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
		kp = ks.K′[jk, jz]

		for (ja, av) in enumerate(gr.a), (jϵ, ϵv) in enumerate(gr.ϵ)

			yv = CoH(av, ϵv, rv, wv)

			res = Optim.optimize(
				ap -> -value(ks, ap, yv, jϵ, jz, kp),
				amin, amax, GoldenSection()
				)

			ap = res.minimizer

			new_v[ja, jϵ, jk, jz] = -res.minimum
			new_a[ja, jϵ, jk, jz] = ap

			new_c[ja, jϵ, jk, jz] = budget_constraint(yv, ap)
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

function update_vϕ!(ks::KS, new_v, new_ϕ; upd_η=0.5)
	ks.vf = ks.vf + upd_η * (new_v - ks.vf)
	# ks.ϕ  = ks.ϕ  + upd_η * (new_ϕ - ks.ϕ )
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

		update_vϕ!(ks, new_v, new_ϕ, upd_η = upd_η)
	end
	return dist
end

function update_K!(ks::KS)
end

function update_prices!(ks::KS, upd_η)
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

		new_w[jk, jz] = (1-α) * zv * (kv/Lv)^α
		new_r[jk, jz] =   α   * zv * (kv/Lv)^(α-1)
	end

	dist_w = sum((new_w - ks.w)^2) / sum((ks.w).^2)
	dist_r = sum((new_r - ks.r)^2) / sum((ks.r).^2)

	dist = max(dist_w, dist_r)

	ks.w = ks.w + upd_η * (new_w - ks.w)
	ks.r = ks.r + upd_η * (new_r - ks.r)

	return dist
end

function eqm!(ks::KS; maxiter::Int64=250, tol::Float64=1e-4)

	dist = 1+tol
	iter = 0

	tol_vfi = 5e-2

	while dist > tol && iter < maxiter
		iter += 1

		dist_p = update_prices!(ks)

		dist_v = vfi!(ks, tol_vfi = tol_vfi)

		dist_K = update_K!(ks)

		dist = max(dist_v, dist_K, dist_p)

		tol_vfi = max(0.95*tol_vfi, 1e-6)
	end
end
