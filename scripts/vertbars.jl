include("../src/bilevelfilterlearning.jl")
include("init_options.jl")
using Plots
using Random: seed!, randn, rand
using MIRTjim: jim
#using Serialization

using Dates
mynow = (i="") -> display((isempty(i) ? "" : string(i)*": ")*
	Dates.format(now(), "HH:MM") ) # handy function to quickly print the time

################################################################################
# set-up parameters
σ = 25/255          	# noise variance (image scale is 0-1)

# these parameters are part of the lower-level denoiser
reg = reg_cr1n!(0.01)	# this is phi (φ) in the code
T = 50 					# maximum number of lower level iterations
# lower level convergences if the norm(∇𝚽) < 1e-10
checkconvergence = s -> _checkconvergence_gnorm(s; gnorm=1e-10)
mom = :pogm 			# momentum setting (:pgm, :fpgm, or :pogm)

# these parameters are part of the upper-level HPO search strategy
niter = 4000 			# maximum number of upper-level iterations
ℓ = initℓ()				# default loss is the squared error loss function

# other settings
c = :civiridis 			# colormap for plotting

# helper function for visualization
function frame_heatmap(x; lc=:red)
	# the framewtyle=:box adds an extra half pixel, so make the frame manually
	m,n = size(x)
	plot!([0.5,0.5],[0.5,m+.5];lc) # left side
	plot!([0.5,n+.5],[0.5,0.5];lc) # bottom
	plot!([n+.5,n+.5],[0.5,m+.5];lc) # right side
	plot!([0.5,n+.5],[m+.5,m+.5];lc) # top
end

################################################################################
# image with vertical bars (bars appear horizontal because of transpose in jim)
xtrue = zeros(32,32); xtrue[:,1:4:end] .= 1
xtrue .-= sum(xtrue[:])/length(xtrue) # make it a zero-mean signal
gr(label="")
function myheatmap(x; c=:grays, frame=true)
	heatmap(x; c=c, aspect_ratio=:equal, border=:none, size=(300,300))
	plot!(clims=(-.25,.75))
	if frame; frame_heatmap(x); end
end
myheatmap(xtrue)

# for reporting quality
myloss = x -> 10*log10(norm(xtrue)^2/norm(x-xtrue)^2) # SNR

################################################################################
# set-up
seed!(0)
𝚽 = init𝚽(xtrue; reg)				 # defines lower-level as denoising cost fcn
y = xtrue + σ*randn(size(xtrue))	# noisy training data

# optimize the lower-level cost and return the loss function
function evaluate_loss(γ; x0 = copy(y))
	xhat,_,gmopt = denoise_caol(y, γ, 𝚽;
		niter=Int(10e3), mom=:pogm, checkconvergence)
	if !gmopt.is_converged
		@warn "Did not reach convergence"
	end
	return norm(xhat-xtrue)/norm(xtrue), xhat
end

# this function records information for later visualization
function userfun(s, l, llres)
	if s.iter >=0 && isa(llres, pogm_state)
		# what to record for every training sample every upper-level iteration
		s.opt["llconv"][s.iter+1] = norm(llres.Fgrad)
		s.opt["llitrs"][s.iter+1] = llres.iter
		s.opt["lossj"][s.iter+1] = norm(llres.xnew-s.xtrue[l])/norm(s.xtrue[l])
	end
	if s.iter >= 1 && l > s.L
		# what to record every upper-level iteration after looping over
		# all the training samples
		s.opt["deltah"][s.iter] = norm(s.γold.h - s.γ.h)/norm(s.γ.h)
		s.opt["deltaβ"][s.iter] = norm(s.γold.β - s.γ.β)/norm(s.γ.β)
		s.out[s.iter] = copyγ(s.γ)
		if mod(s.iter,500)==0; mynow(s.iter); end # print time every 500 iters
	end
	if s.iter == 0 && l>0 && l <= s.L
		# initialization going into the first iteration
		copy!(s.opt["cgres"][l], s.xhat[l]) #
	end
end
# struct to hold all the optional information during the bilevel method
opt = (niter, K) -> Dict{String,Any}("h"=>[Flux.ADAM() for k=1:K], "β"=>Flux.ADAM(),
						"cgres" => [zeros(size(xtrue))],
						"llconv" => zeros(niter), "llitrs" => zeros(niter),
						"lossj" => zeros(niter), "true_lossj" => zeros(niter),
						"deltah" => zeros(niter), "deltaβ" => zeros(niter))

################################################################################
# learn 4 random filters, each size 4x2
seed!(0)
γ = initγ(;h=[normalize(randn(4,2)) for _=1:4])
β0s = -6:.1:-4
loss = zeros(length(β0s))
for (i,β0) in zip(1:length(β0s), β0s)
	loss[i] = evaluate_loss(initγ(;h=γ.h, β=β0))[1]
end
β0 = β0s[argmin(loss)]

γ = initγ(;h=γ.h, β=β0)
init_loss, xhatinit = evaluate_loss(γ)
state = initializebilevelstate([xtrue], [y], γ; niter, fun=userfun, 𝚽,
	opt=opt(niter, γ.K),
	lloptimizer=(s::BilevelState, l::Int) -> denoise_caol!(gm_Φ(s.y[l], s.γ,
				s.𝚽; x0=s.xhat[l], niter=T, sb=s.sb[l], mom,
				checkconvergence))[end])
state.xhat[1] .= xhatinit
state.opt["initγ"] = copyγ(γ)
bilevel_learn_H!(state)

plot(log.(state.opt["lossj"]))
K = length(state.γ.h)
effectiveβ = [norm(state.γ.h[k])*exp(state.γ.β[k]+state.γ.β[0]) for k=1:K]

# circshift for easier interpretation (doesn't change filter meaning)
for (k,shift) in zip([1,2,3],[(1,1),(0,1),(1,0)])
	state.γ.h[k] = setoffsets(circshift(state.γ.h[k].parent, shift))
	state.opt["initγ"].h[k] =
		setoffsets(circshift(state.opt["initγ"].h[k].parent, shift))
end
validateγ!(state.γ) 	# makes hrev match the shifted version of h
validateγ!(state.opt["initγ"])
pout = [heatmap(normalize(state.γ.h[k].parent), border=:none, c=:cividis,
			xticks=[],yticks=[], axis_ratio=:equal, size=(200,100),
			title=string(round(effectiveβ[k];digits=3)),
			xlabel="", clims=(-.5,.5), colorbar=false)
			for k in sortperm(effectiveβ; rev=true)]
plot(pout..., layout=(1,4), size=(600,300))
#savefig("vertbars\\learnedfromnoise_4random4x2filters.png")

pinit = [heatmap(normalize(state.opt["initγ"].h[k].parent), border=:none,
			c=:cividis, xticks=[],yticks=[], axis_ratio=:equal, size=(200,100),
			title=" ", xlabel="", clims=(-.5,.5), colorbar=false)
			for k in sortperm(effectiveβ; rev=true)]
plot(pinit..., layout=(1,4), size=(600,300))
#savefig("vertbars\\learnedfromnoise_4random4x2filters_init.png")

display(vcat([sum(hk; dims=1) for hk in state.γ.h]...)) # column sums
display(hcat([sum(hk; dims=2) for hk in state.γ.h]...)) # row sums
final_loss, xhat = evaluate_loss(state.γ; x0 = copy(state.xhat[1]))
myheatmap(xhat); title!(string(round(myloss(xhat); digits=2)))
#savefig("vertbars\\vertbars_xhat_4random4x2filters.png")

scatter(1:size(xhat,2),xhat[16,:], mc=:blue, size=(300,100))
plot!(xhat[16,:], lw=0.5, lc=:blue)
#savefig("vertbars\\vertbars_xhat_4random4x2filters_slice.png")


################################################################################
# repeat the above but now initializing with 3x3 DCT filters
filterdims = (3,3) 	# will make 3x3 filters
h = getDCTfilters(;filterdims, dc=false) 	# non-DC DCT filters

# do a quick grid search to initialize β to a reasonable value
K = length(h)
β0s = -4:.1:-2
loss = zeros(length(β0s))
for (i,β0) in zip(1:length(β0s), β0s)
	loss[i] = evaluate_loss(initγ(;h=deepcopy(h),β=β0))[1]
end
β0 = β0s[argmin(loss)]

γ = initγ(;h=deepcopy(h),β=β0)
initloss, xhatinit = evaluate_loss(γ; x0 = copy(y))[2]
# learn γ from the DCT initialization
state = initializebilevelstate([xtrue], [y], γ; niter, fun=userfun, 𝚽,
	opt=opt(niter, γ.K),
	lloptimizer=(s::BilevelState, l::Int) -> denoise_caol!(gm_Φ(s.y[l], s.γ,
				s.𝚽; x0=s.xhat[l], niter=T, sb=s.sb[l], mom,
				checkconvergence))[end])
state.xhat[1] .= xhatinit
state.opt["initγ"] = copyγ(γ)
bilevel_learn_H!(state)
plot(state.opt["lossj"], lc=:black, border=1,
	xlabel="iterations", ylabel="norm(xtil(γ^{(u)})-xtrue)/norm(xtrue)")

# now get the true loss at each 10th γ^{(u)} value
xhat = copy(y)
inds = 1:10:state.niter
for u in inds
	# possibly causing ambiguity and report error on my local machine
	state.opt["true_lossj"][u], xhat_ambiguos = evaluate_loss(state.out[u]; x0=xhat)
end
# plot the true and estimated upper-level loss function
scatter!(inds, state.opt["true_lossj"][inds], mc=:blue, ms=2)

plot(log.(state.opt["deltah"])) # relative norm of h change function
plot!(xlabel="Iterations", ylabel="log(norm(h-h_old)/norm(h))")

# plot the filters in order of norm(hk)*βk
pinit = Array{Any}(undef,length(h))
pout = Array{Any}(undef,length(h))
effectiveβ = [norm(state.γ.h[k])*exp(state.γ.β[k]+state.γ.β[0]) for k=1:K]
pout = [heatmap(normalize(state.γ.h[k].parent), border=:none, c=:cividis,
			xticks=[],yticks=[], axis_ratio=:equal, size=(200,100),
			title=string(round(effectiveβ[k];digits=3)),
			xlabel="", clims=(-.5,.5), colorbar=false)
			for k in sortperm(effectiveβ; rev=true)]
pinit = [heatmap(normalize(state.opt["initγ"].h[k].parent), border=:none,
			c=:cividis, xticks=[],yticks=[], axis_ratio=:equal, size=(200,100),
			title=" ", xlabel="", clims=(-.5,.5), colorbar=false)
			for k in sortperm(effectiveβ; rev=true)]
p1 = plot(pinit[:]..., layout=(1,8), size=(1000,100))
p2 = plot(pout[:]..., layout=(1,8), size=(1000,100))
plot(p1,p2,layout=(2,1), size=(1000,300))
#savefig("vertbars_DCT_learnedh.png")

# look at denoised output
xhat_hinit = evaluate_loss(initγ(;h=deepcopy(h), β=β0))[2]
myheatmap(xhat_hinit)
title!(string(round(myloss(xhat_hinit);digits=3)))
#savefig("vertbars\\vertbars_xhat_hinit.png")

xhat_hout = evaluate_loss(state.γ)[2]
myheatmap(xhat_hout)
title!(string(round(myloss(xhat_hout);digits=3)))
#savefig("vertbars\\vertbars_xhat_hout.png")

myheatmap(y)
title!(string(round(myloss(y);digits=2)))
#savefig("vertbars\\vertbars_y.png")
scatter(1:size(y,2),y[16,:], mc=:blue, size=(300,100))
plot!(y[16,:], lw=0.5, lc=:blue)
#scatter!(1:size(xtrue,2),xtrue[16,:],mc=:black)
#savefig("vertbars\\vertbars_y_slice.png")

myheatmap(xtrue)
#savefig("vertbars\\vertbars_xtrue")
