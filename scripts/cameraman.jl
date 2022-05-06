using TestImages, Images, ImageMagick
using Plots

include("../src/bilevelfilterlearning.jl")
include("init_options.jl")
using Random: seed!, randn, rand
using MIRTjim: jim		# jim = jiffy image (display)
#using Serialization		# save results
using BM3DDenoise 		# comparison denoising method

using Dates
mynow = (i="") -> display((isempty(i) ? "" : string(i)*": ")*
						Dates.format(now(), "HH:MM") )

################################################################################
seed!(0) 			# for reproducability

# set-up parameters
datadir = "C:/Users/Caroline/Desktop/BSDS500/images/"
ntrain = 3          # number of training images to use
nx, ny = (50,50) 	# patch size
σ = 25/255  		# standard deviation of noise (image scale is 0-1)
ℓ = initℓ()			# default is MSE loss function
recordγ = 50		# number of iterations between saving γ^(u) for later use

# set up and visualize the initial filters
filterdims = (7,7) 	# will make 3x3 filters
h = getDCTfilters(;filterdims, dc=false)
jim(cat(h...,dims=3)); title!("Initial filters")

# these parameters are part of the lower-level denoiser
ϵ = 1e-5/sqrt(nx*ny) # convergence threshold (relative change in ||x||)
reg = reg_cr1n!(0.01) # you could also try: reg_gm(0.1)
checkconvergence = s->_checkconvergence_gnorm(s;gnorm=ϵ) 
T = 10				# number of lower-level iterations
mom = :pogm			# lower-level gradient method

# these parameters are part of the upper-level HPO search strategy
niter = 50 			# upper-level iterations U. This is 50000 in paper

################################################################################
# use Julia's built-in test images
#sfull = [convert(Array{Float64},channelview(testimage(i))[1,:,:]) for i in
#	["house", "jetplane"]]
sfull = [convert(Array{Float64},testimage("cameraman"))]

# hand-pick some training patches that have some structure to them
L = length(sfull)
inds1 = [311:(311+nx-1),221:(221+ny-1)]
jim(hcat([sfull[i][inds1...] for i=1:L]...)', clims=(0,1), colorbar=false,
	xticks=[], yticks=[], border=:none, size=(200,200), axis_ratio=:equal)
inds2 = [221:(221+nx-1),371:(371+ny-1)]
jim(hcat([sfull[i][inds2...] for i=1:L]...)', clims=(0,1), colorbar=false,
	xticks=[], yticks=[], border=:none, size=(200,200), axis_ratio=:equal)
inds3 = [101:(101+nx-1),251:(251+ny-1)]
jim(hcat([sfull[i][inds3...].*1/.9 for i=1:L]...)', clims=(0,1),
	xticks=[], yticks=[], border=:none, size=(200,200), axis_ratio=:equal)
#xtrue = [[sfull[i][inds1...] for i=1:L]..., [sfull[i][inds2...] for i=1:L]...]
#jim(cat(s'..., dims=3))
xtrue = [[sfull[1][i...] for i in (inds1, inds2, inds3)]...]
L = length(xtrue)

# add noise
seed!(0)
y = [xtrue[n] + σ*randn(size(xtrue[n])) for n=1:length(xtrue)]
n = 1; jim(cat(xtrue[n]', y[n]', dims=3))

# use full cameraman image as the test image
seed!(1)
xtest = sfull[1]
ytest = xtest + σ*randn(size(xtest))
myloss = (x;xt=xtest) -> string(round(20*log10(norm(xt)/norm(x-xt)); digits=3))

# set-up some more variables
𝚽 = init𝚽(xtrue[1]; reg, opt=:denoising) # lower-level cost function
sb = [scratch(xtrue[l], h; y=y[l]) for l=1:length(xtrue)]
sb2 = [scratch(xtrue[l], h; y=y[l]) for l=1:length(xtrue)]

# do a quick grid search for β0 first to get a reasonable initialization
function evaluate_loss(γ)
	L = length(xtrue)
	xhat = [denoise_caol(y[l],γ,𝚽; niter=Int(10e4),
		mom=:pogm, checkconvergence)[1] for l=1:L]
	return sum([ℓ.loss(xhat[l], xtrue[l], γ) for l=1:L])/L,
			1/L * sum([norm(xhat[l]-xtrue[l])/norm(xtrue[l]) for l=1:L]),
			xhat
end
β0s = -5:1:0
loss = zeros(length(β0s))
for (i,β0) in zip(1:length(β0s), β0s)
	loss[i] = evaluate_loss(initγ(;h=deepcopy(h),β=β0))[1]
end
β0 = β0s[argmin(loss)]
@show β0
γ = initγ(;h=deepcopy(h),β=β0)
init_loss, xhat_init = evaluate_loss(γ)[2:end]

# this runs during the bilevel optimization to record information
function userfun(s::BilevelState, l, llres)
	t = now()
	if l > s.L && s.iter >= 1
		# what to record every upper-level iteration after looping over
		# all the training samples
		if mod(s.iter,recordγ) == 0
			s.out[s.iter] = copyγ(s.γ)
		end
		if mod(s.iter,1000) == 0
			mynow(s.iter)
			@show s.fcost
		end
		s.opt["telapsed"][s.iter] = Dates.value(t-s.opt["tstart"]-
									s.opt["userfuntime"])
		s.opt["loss"][s.iter] = s.fcost
	elseif s.iter >=0 && isa(llres, pogm_state)
		# what to record for every training sample every upper-level iteration
		s.opt["llconv"][s.iter+1, l] = norm(llres.Fgrad)
		s.opt["llitrs"][s.iter+1, l] = llres.iter
		s.opt["lossj"][s.iter+1, l] = norm(llres.xnew - s.xtrue[l])/
										norm(s.xtrue[l])
		# can recover the loss function by
		# sum((mystate.opt["lossj"][s.iter+1,:] .* norm.(xtrue)).^2 /2 )/L
	end
	if s.iter == 0 && l>0 && l <= s.L
		# initialization going into the first iteration
		copy!(s.opt["cgres"][l], s.xhat[l])
	end
	s.opt["userfuntime"] += now()-t # exclude this function time in eval time
end

# function to save memory by removing fields we don't need to reference later
function removefields(s::BilevelState)
	s.y = []
	s.sb = []
	s.opt["cgres"] = []
	s.xtrue = []
	s.xhat = []
	# remove the learned paramters per iteration except the last
	s.out = s.out[s.niter]
	return s
end

function run_bilevel_learn_H_ADAM(niter, T, mom;
	α𝚽=[],	γ = initγ(;h=deepcopy(h),β=β0),
	xhat = deepcopy(xhat_init), descend_h=true)

	userfuntime = now()
	userfuntime -= userfuntime # a roundabout way of making a millisecond object
	opt = Dict{String,Any}("h"=>[Flux.ADAM() for k=1:length(γ.h)],
		"β"=>Flux.ADAM(),
		"cgres" => [zeros(size(xtrue[l])) for l=1:length(xtrue)],
		"tstart" => now(), "userfuntime" => userfuntime,
		"llconv" => zeros(Float32, niter, length(xtrue)),
		"llitrs" => zeros(Float32, niter, length(xtrue)),
		"lossj" => zeros(Float32, niter, length(xtrue)),
		"telapsed" => zeros(Float32, niter), # will be in milliseconds
		"loss" => zeros(Float32, niter))
	fill!(opt["llconv"], -1)
	mystate = initializebilevelstate(xtrue, y, γ;
		niter, fun=userfun, 𝚽, ℓ, xhat, descend_h, opt,
		lloptimizer=(s::BilevelState,l::Int) -> denoise_caol!(gm_Φ(y[l],s.γ,s.𝚽;
					x0=s.xhat[l], niter=T, sb=s.sb[l], fun=gm_user_fun,
					alpha=α𝚽, mom, restart=:gr, checkconvergence))[end],
		dℓdγ = (s::BilevelState, l) -> ift_gradient(s.xhat[l],xtrue[l],y[l],s.γ,
				s.𝚽,s.ℓ; sb=s.sb[l], sb2=sb2[l], CG_init=s.opt["cgres"][l]),
		updateγ! = (s) -> updateγ_ADAM!(s.γ, s.opt["h"], s.opt["β"], s.dh, s.dβ;
				s.descend_h, s.descend_β))
	mystate.opt["tstart"] = now()
	bilevel_learn_H!(mystate)

	return mystate
end
run_bilevel_learn_H_ADAM(2, 2, :pogm) # compile

#
function continue_learn_H_ADAM(state, niter)
	state.niter = state.niter + niter
	L = length(xtrue)
	state.opt["llconv"] = vcat(state.opt["llconv"], zeros(Float32, niter, L))
	state.opt["llitrs"] = vcat(state.opt["llitrs"], zeros(Float32, niter, L))
	state.opt["lossj"] = vcat(state.opt["lossj"], zeros(Float32, niter, L))
	state.opt["telapsed"] = vcat(state.opt["telapsed"], zeros(Float32, niter))
	state.opt["loss"] = vcat(state.opt["loss"], zeros(Float32, niter))
	state.opt["tstart"] = now() # TODO - meaningless!
	state.opt["previousout"] = state.out
	state.out = similar(Array{Any}, state.niter+1)
	bilevel_learn_H!(state)
end


################################################################################
# learn only beta first

# learn the tuning parameters
#state = deserialize("cameraman\\state_cameraman_betaonly.dat")
# this takes about 15 minutes to run 500 iterations on my laptop
state = run_bilevel_learn_H_ADAM(niter, T, mom; descend_h=false)
#serialize("cameraman\\state_cameraman_betaonly.dat", state)
plot(state.opt["loss"])

# see how well the learned betas denoise one of the training patches
l = 1 # which training patch to look at
γ0 = state.γ; validateγ!(γ0)
xhat, f, gmopt = denoise_caol(y[l], γ0, state.𝚽; niter=10000, checkconvergence,
	x0=deepcopy(state.xhat[l]), fun=s->s.out[s.iter+1]=norm(s.Fgrad))
jim(hcat(xtrue[l], y[l], gmopt.xnew)')

# see how well they denoise the full image
niter_denoising = 100 # TODO - this is not enough to reach convergence
𝚽full = init𝚽(xtest; reg)
xhat_full, f, gmopt = denoise_caol(ytest, state.γ, 𝚽full; niter=niter_denoising,
	checkconvergence, fun=s->s.out[s.iter+1]=s.fcostnew)
@show gmopt.iter
jim(xhat_full', title="beta only ("*myloss(xhat_full)*" dB)", xticks=[],
	yticks=[], border=:none, colorbar=false, clims=(0,1), size=(400,420))
jim(ytest', title="noisy image ("*myloss(ytest)*" dB)", xticks=[],
	yticks=[], border=:none, colorbar=false, clims=(0,1), size=(400,420))

# look at the filters
γ = state.γ
h = γ.h
K = length(h)
maxval = maximum([maximum(abs.(h[k] .* exp.(γ.β[0]+γ.β[k]))) for k=1:K])
p = Array{Any}(undef,K+1)
p[1] = heatmap(0. .*normalize(h[1].parent.*exp(γ.β[0]+γ.β[1])), # ignore DC
	c=:grays, axis_ratio=:equal, border=:none, colorbar=false, size=(100,100))
inds = 1:K 	# plot in DCT order (use next line for plotting in magnitude order)
#inds = sortperm([norm(h[k].parent.*exp(γ.β[k])) for k=1:K]; rev=true)
for (i,k) in zip(1:K,inds)
	p[i+1] = heatmap(h[k].parent.*exp(γ.β[0]+γ.β[k]), clims=(-maxval,maxval),
		c=:cividis, axis_ratio=:equal, border=:none, xticks=[], yticks=[])
	plot!(xlabel="", size=(100,100), colorbar=false, titlefontsize=12)
	e = exp(γ.β[k])*norm(γ.h[k])
	title!(string(round(e; digits= e>0.01 ? 2 : 3)))
end
plot(p...,layout=(7,7), size=(1000,1000), margin=0Plots.mm)
# savefig("cameraman\\dct_betaandh_7x7_withtitles.png")


################################################################################
# learn h and beta (this takes quite a while to run on a laptop)
#state = deserialize("cameraman\\state_cameraman.dat")
# this takes about TODO minutes to run 500 iterations on my laptop
state = run_bilevel_learn_H_ADAM(niter, T, mom)
#serialize("state_cameraman.dat", state)

# plot the upper-level loss function
pcost = plot(state.opt["loss"], lw=2, lc=:blue, label="h and beta",
	xlabel="iterations", ylabel="UL loss", size=(300,200))

γ = state.out[50*argmin([state.opt["loss"][i*50] for i=1:Int(state.niter/50)])]
pcosttime = plot(state.opt["telapsed"]./1000 ./60^2, state.opt["loss"], lw=2,
	lc=:blue, label="h and beta", xlabel="time elapsed (hr)", size=(300,200))
pllconv = plot(log10.(mean(state.opt["llconv"], dims=2)/sqrt(length(xtrue[1]))),
	xlabel="iteration", ylabel="LL grad norm (mean)", lc=:blue, size=(300,200))

# see how well they denoise the full image
𝚽full = init𝚽(sfull[1]; reg)
xhat_full, f, gmopt = denoise_caol(ytest, state.γ, 𝚽full; niter=niter_denoising,
	checkconvergence, fun=s->s.out[s.iter+1]=s.fcostnew)
jim(xhat_full', title="h and beta ("*myloss(xhat_full)*" dB)", xticks=[],
	yticks=[], border=:none, colorbar=false, clims=(0,1), size=(400,420))
@show gmopt.iter

# look at the filters
h = state.γ.h
K = length(h)
maxval = maximum([maximum(abs.(h[k] .* exp.(γ.β[0]+γ.β[k]))) for k=1:K])
p = Array{Any}(undef,K+1)
p[1] = heatmap(0. .*normalize(h[1].parent.*exp(γ.β[0]+γ.β[1])), # ignore DC
	c=:grays, axis_ratio=:equal, border=:none, colorbar=false, size=(100,100))
inds = 1:K 	# plot in DCT order (use next line for plotting in magnitude order)
#inds = sortperm([norm(h[k].parent.*exp(γ.β[k])) for k=1:K]; rev=true)
for (i,k) in zip(1:K,inds)
	# use commented out line insead of the next one to plot normalized filters
	#p[i+1] = heatmap(h[k].parent.*exp(γ.β[0]+γ.β[k]), clims=(-maxval,maxval),
	p[i+1] = heatmap(normalize(h[k].parent.*exp(γ.β[0]+γ.β[k])),
		c=:cividis, axis_ratio=:equal, border=:none, xticks=[], yticks=[])
	plot!(xlabel="", size=(100,100), colorbar=false, titlefontsize=12)
	e = exp(γ.β[k])*norm(γ.h[k])
	title!(string(round(e; digits= e>0.01 ? 2 : 3)))
end
plot(p...,layout=(7,7), size=(1000,1000), margin=0Plots.mm)
#savefig("cameraman\\dct_betaandh_7x7_withtitles.png")

################################################################################
# Compare to BM3D
res = bm3d(ytest, σ)
jim(res', title="BM3D ("*myloss(res)*" dB)", xticks=[], yticks=[], border=:none,
	colorbar=false, clims=(0,1), size=(400,420))
#savefig("cameraman\\camera_xhat_bm3d.png")
