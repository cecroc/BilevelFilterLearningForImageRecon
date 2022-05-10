include("systemmatrix.jl")
#include("gm_restart.jl")
include("regularizers.jl")
include("inplacefunctions.jl")

using Pkg
Pkg.add(url="https://github.com/cecroc/MIRT.jl")  # TODO: merge with MIRT branch
using MIRT #: new_pogm_state, pogm_restart!, reinit_pogm_state!, pogm_state

using LinearAlgebra
using LinearMaps
using ImageFiltering # provides the convolution function
using OffsetArrays 	# all filters will be OffsetArrays to preserve indices
# TODO: Flux is the packages causing the warning about OffsetArrays...
import Flux # import (not `using') to avoid σ name conflict!
using InplaceOps
import IterativeSolvers: cg!, CGStateVariables

#=
top level:
    argmin_(γ) ℓ(γ; X)
    ℓ(γ; X) = ∑_l 1/2||xtrue_l - ̂x(s_l+n, γ)||^2_2 "loss function"
lower level:
    ̂x(y,γ) = argmin_x 𝚽(x;y,γ)
    𝜱(x;y,γ) = 1/2||Ax-y||^2_2 + exp(β_0) R(x;γ)       "cost function"
    R(x;γ) = ∑_k exp(β_k) 1'φ(h_k ⊛ x; ϵ)              "regularizer"
Here, γ can be a vector of any set of parameters, such as hk, α, β...
=#

# padding functions
getpad(h) = Pad(:circular,
            Tuple(Int.(ceil.( maximum(hcat([collect(size(h[i])) for i=1:length(h)]...)',dims=1) ./2))),
            Tuple(Int.(ceil.( maximum(hcat([collect(size(h[i])) for i=1:length(h)]...)',dims=1) ./2))))
padxj(xj,h) = padarray(xj, getpad(h))
function padxj!(res, xj::AbstractArray, padval)
    # this allocates some memory, but it's fairly efficient and it's easy to read
    ba = BorderArray(xj, padval)
    ##res .= ba # equivalent to copy!(res, ba)
	copy!(res, ba)
end
function padx!(res,y,h; padval=getpad(h))
    for l = 1:length(y)
        padxj!(res[l],y[l],padval)
    end
end

# functions to create and reverse h with the correct indeces
"""
Takes in a single n-dimensional matrix (a filter) and returns the OffsetArray
for the zero-centered filter.

Examples:
- ``setoffsets([1,-1])`` returns an OffsetArray with indices 0:1
- ``setoffsets(zeros(3,3))`` returns an OffsetArray with indices -1:1 × -1:1
"""
setoffsets(hk) = OffsetArray(hk, Tuple([-Int(ceil((N-2)/2)):Int(floor(N/2))
				for N in size(hk)]))
function revhk(h)
    for d = 1:ndims(h)
        h = reverse(h, dims=d)
    end
    return OffsetArray(h, Tuple([-Int(floor(N/2)):Int(ceil((N-2)/2)) for N in size(h)]) )
end
revh = (h) -> [revhk(hk) for hk in h]

# in-place direct convolution function
conv!(zjk, xjpad, hk) = imfilter!(zjk,xjpad,(hk,),NoPad(),Algorithm.FIR())

# set the offsets for a tuning parameter based on the number of filters (K)
function _getvalidβ(β,K)
    if length(β) == 1;
        β = [β; zeros(K)];
    elseif length(β) == K
		β = [0; β]
	else
        @assert length(β) == K+1 "β must be of length 1, K, or K+1"
    end
    β = OffsetArray(β,0:K)
    return β
end

# structure to hold pointers to temporary variables
# (these are arrays we only want to allocate once)
struct ScratchBuffer
    x1
    x2
    x3
    xpad1
    xpad2
    padval
	y1
	y2
end
scratch(x,h;y=x) = ScratchBuffer(deepcopy(x), deepcopy(x), deepcopy(x),
                        padxj(x,h), padxj(x,h), getpad(h),
						deepcopy(y), deepcopy(y))

mutable struct Hyperparameters
	h::Vector 			# Array of OffsetArrays (length K)
	β::OffsetArray		# vector of tuning parameters (indices [0..K])
	α::Vector			# vector of coefficient vectors (length K)
	B::Vector 			# Vector of basis elements (length K)
	hrev
	K::Int
end
initγ(; h=[], β=0, α=[], B=[]) = Hyperparameters([setoffsets(hk) for hk in h],
	_getvalidβ(β,length(h)), α, B, revh(h), length(h))
function validateγ!(γ) # TODO generalize for α, B option
	γ.K = length(γ.h)
	γ.hrev = revh(γ.h)
	@assert length(γ.β) == γ.K + 1
end
function copyγ(γ)
	return initγ(;h=[deepcopy(hk) for hk in γ.h],
		β=deepcopy(γ.β), α=deepcopy([γ.αk for αk in γ.α]),
		B=deepcopy([γ.Bk for Bk in γ.B]))
end

# TODO - code does not currently support non-zero ∇γloss
# would need to correct dimensions throughout for that
# and probably define ∇hkloss, ∇βkloss, and ∇αkloss separately
struct LossFunction
	loss::Function 		# (xhat, xtrue, γ) -> loss function value
	∇xloss::Function 	# (xhat, xtrue, γ) -> ∇_x l(x,γ) evaluated at xhat, s, γ
	∇xloss!::Function 	# in-place version with inputs: (res, xhat, xtrue, γ)
	∇γloss::Function  	# (xhat, xtrue, γ) -> ∇_γ l(x,γ) evaluated at xhat, s, γ
end
function _∇xloss!(dloss,xhat,xtrue) # in-place version of d_loss
    minus!(dloss, xhat, xtrue)
end
initℓ(; loss = (xhat,xtrue,γ)-> 1/2*sosdiff(xhat,xtrue),
		∇xloss = (xhat,xtrue,γ) -> xhat-xtrue,
		∇xloss! = (res,xhat,xtrue,γ) -> minus!(res, xhat, xtrue),
		∇γloss = (xhat,xtrue,γ) -> 0) =
		LossFunction(loss, ∇xloss, ∇xloss!, ∇γloss)

# TODO - could add a term that only takes γ or add γ to the data-fit term
# and give it a new name. Then would need to modify _d2Φdxdh!
struct CostFunction
	datafit::Function # (x,y;Ax) -> data-fit cost (real, non-neg)
	ddatafit::Function # (x,y;Ax) -> d(datafit)/dx
	ddatafit!::Function # (res,x,y;Ax) in-place version of ddatafit
	A 	# System matrix (LinearMap or Matrix)
	At 	# Adjoint system matrix for A' operation
	LC 	# a Lipschitz constant for the data-fit term
	reg::Regularizer!
end
# standard 2-norm data-fit term
function _datafit(x, y, A; Ax=[]) # 1/2 || Ax-y ||_2^2
	if isempty(Ax)
        Ax = A*vec(x)
    end
	return 1/2*sosdiff(Ax,y)
end
# derivative of datafit for common 2-norm data-fit term
function _ddatafit!(res, x, y, A, At; Ax=[], ytemp=similar(y))
    if isempty(Ax)
        Ax = A*vec(x)
    end
	if isempty(ytemp)
		ytemp = similar(y)
	end

	minus!(ytemp, Ax, y) #  Ax-y
	mul!(res, At, vec(ytemp)) # A'(Ax-y)

	return res
end
_init𝚽(x, reg, AAt; datafit  = (x,y;Ax=[])->_datafit(x,y,AAt[1];Ax),
		 ddatafit = (x,y)->_ddatafit!(similar(x),x,y,AAt[1],AAt[2]),
		 ddatafit! = (res,x,y;Ax=[],xtemp=[],ytemp=similar(y)) ->
		 			_ddatafit!(res,x,y,AAt[1],AAt[2];Ax,ytemp)) =
	CostFunction(datafit, ddatafit, ddatafit!, AAt[1], AAt[2], AAt[3], reg)
init𝚽(x; reg=reg_2norm!(), kwargs...) = _init𝚽(x, reg, defineA(; xdim=size(x), values(kwargs)...))

"""
Given a set of basis vectors and coefficients, form the linear combination
    Input:
        B   :   List of length K. Each B[k] is a basis vector.
        α   :   Linear combination coefficients, also of length K.
    Output:
        hk  :   Vector of the linear combinations. hk = ∑_k α[k]*B[k]
"""
_hkfromαk = (Bk,αk) -> sum([αk[i]*Bk[i] for i=1:length(αk)],dims=1)
"""
Given a set of basis vectors and coefficients, form the linear combination
    Input:
        B   :   List of length i. Each B[i] is a list of k basis vectors.
        α   :   List of length i. Each α[i] is a list of k linear coefficients.
    Output:
        h   :   List of the linear combinations. h[i] = ∑_k α[i][k]*B[i][k]
"""
hfromα = (γ) -> [_hkfromα(γ.B[k],γ.α[k]) for k=1:γ.K]

"""
Find the Lipschitz constant for the lower-level regularizer
"""
regLC(γ,reg) = exp(γ.β[0])*reg.Ldφ*sum([exp(γ.β[k])*sum(abs.(γ.h[k])).^2 for k=1:γ.K])

"""
obj = _dRdx!(dx,xpad,h,reg::regularizer;
    sb=scratch(dx,h), β=zeros(length(h)), hrev=revh(h))

 This function finds dR/dx = exp(β0) ∑_k exp(βk) Ck' dφ.(h_k ⊛ x) and stores
    the result in dx (in-place operation). It returns the regularization
    function value: R(x) = exp(β0) ∑_k exp(βk) 1'φ.(hk ⊛ x)

    Input:
        x   :   value at which to evaluate the derivative (n-dim array)
        γ   :   defines filters and tuning parameters
        𝚽  :   includes reg field, a regularizer struct that defines φ and dφ
    Output:
        obj :   R(x) = exp(β0) ∑_k exp(βk) 1'φ.(h_k ⊛ x)

	This function still allocates some memory (such as h[k]), and the overall
		memory use will grow with K. But, it should not grow with size(x)
		and the maximum memory use at any time is pretty small.
"""
function _dRdx!(dx, x, γ::Hyperparameters, 𝚽::CostFunction; sb=scratch(dx,γ.h))
	# set-up
	xpad = padxj!(sb.xpad2, x, sb.padval)

    # rename buffers for easier reading (zero cost)
    hconvx = sb.x1
    buf = sb.x2
    bufpad = sb.xpad1

    obj = 0.0
    fill!(dx,0)    # initialize output gradient to 0
    for k in 1:γ.K
		eβk = exp(γ.β[k])
        conv!(hconvx,xpad,γ.h[k])     	# hk ⊛ x
        copy!(buf, hconvx)            	# buffer to hold hk ⊛ x
        𝚽.reg.φ(hconvx);
		obj += eβk*sum(hconvx) 	# obj += exp(βk)*sum(φ.(hk ⊛ x))
        𝚽.reg.dφ(buf) 					# buf = dφ.(hk ⊛ x)
        padxj!(bufpad, buf, sb.padval)
        conv!(hconvx,bufpad,γ.hrev[k]) 	# hconvx = dφ.(hk ⊛ x) ⊛ htil[k]
        lmul!(eβk, hconvx) 		# hconvx .*= eβk;
		plus!(dx, dx, hconvx)
    end
	eβ0 = exp(γ.β[0])
	lmul!(eβ0, dx) # dx .*= exp(γ.β[0])
    return obj*eβ0
end

# This is a helper functions for finding ̂x:
# Stores the gradient at x in fgrad and returns the cost function value
function _f_cost_grad!(fgrad, x, y, γ::Hyperparameters, 𝚽::CostFunction, sb;
	xax=axes(sb.x1))

	# regularizer evaluation and gradient
	x = reshape(x,xax)
    obj = _dRdx!(fgrad,x,γ,𝚽; sb)

	# calculate Ax once then use in both datafit evaluation and ddatafit
	Ax = view(sb.y2, :) #@view view(sb.ypad1, axes(y)...)[:]
	mul!(Ax, 𝚽.A, vec(x)) # Ax = A* vec(x)

	# add data-fit and regularizer gradients and evaluations
	tmp = view(sb.x3, :) # will hold gradient from data-fit term
	𝚽.ddatafit!(tmp, x, y; Ax, xtemp=sb.x1, ytemp=sb.y1)
	plus!(fgrad, tmp, fgrad)
	return 𝚽.datafit(x,y;Ax) + obj # data-fit + regularizer objective
end
function f_grad!(res, x, y, γ::Hyperparameters, 𝚽::CostFunction, s::pogm_state, sb)
	 s.fcostnew = _f_cost_grad!(res, x, y, γ, 𝚽, sb)
end
function gm_user_fun(s::pogm_state)
	if s.iter>0 # set cost function after gradient is computed
		s.out[s.iter] = s.fcostnew
	end
	if s.is_converged # clean up the output
		s.out = s.out[1:s.iter]
	elseif s.iter == s.niter # reached max iterations but did not converge
		s.out = s.out[1:s.iter]
		# can print out a warning here if desired
	end
end
function gm_Φ(y, γ, 𝚽; niter=1000, x0=y, sb=scratch(x0, γ.h; y),
	fun = (s::pogm_state) -> gm_user_fun(s),
	kwargs...) # mom=:pgm, restart=:none...

	@assert size(𝚽.A,2)==length(x0) "Must provide x0 initializzation to gm_Φ "*
	 	"function if size(x)!=size(y)"

	# Not tested with a prox operator or ogm.
	new_pogm_state(;
	    f_cost = (s::pogm_state) -> s.fcostnew, # set in f_grad! function
	    f_grad! = (res, x; pogm_state) -> f_grad!(res, x, y, γ, 𝚽, pogm_state, sb),
		f_L = 𝚽.LC + regLC(γ,𝚽.reg),
	    g_prox! = (res, z, c::Real; pogm_state) -> copy!(res,z), # no prox operator
	    fun = (s::pogm_state) -> fun(s),
		x0, niter,
		values(kwargs)...)
end

"""
xs, fs, rs, out = denoise_caol!(xpad,y,h,β,reg::regularizer; x0=y, gm_opt,
    sb)
 Find ``x̂ = argmin_x 𝚽(x;y,γ)
    given 𝜱(x;y,γ) = 1/2||y-x||^2_2 + exp(β0) R(x;γ)    "cost function"
    and R(x;γ) = ∑_k exp(βk) 1'φ(h_k ⊛ x; ϵ)             "regularizer"
 Input:
    reg is a regularizer struct that defines φ, dφ, and Ldφ
"""
function denoise_caol!(gm_opt::pogm_state)
    pogm_restart!(gm_opt)
    return gm_opt.xnew, gm_opt.out, gm_opt
end
function denoise_caol(y,γ,𝚽; kwargs...)
	return denoise_caol!(gm_Φ(y, γ, 𝚽; kwargs...))
end

"""
dh_ax = f.(z^(ax)) + ̃h ⊛ (df.(z) .* x^(-ax)) where z = h ⊛ x
	and f is 𝚽.dφ
    xax should be axes(x)
"""
function _d2Φdxdh!(dxdh, x, γ, 𝚽, k, xax, sb)
	xpad = sb.xpad2
	hk = γ.h[k]

	padxj!(xpad, x, sb.padval)
    fill!(dxdh,zero(eltype(xpad)))
    z   = sb.x1; conv!(z,xpad,hk)        	# z = h ⊛ x (using x1 memory slot)
    fz  = sb.x2; copy!(fz,z); 𝚽.reg.dφ(fz)     # fz  = dφ(h ⊛ x) (in x2 slot)
    dfz = sb.x3; copy!(dfz,z); 𝚽.reg.ddφ(dfz) # dfz = ddφ(h ⊛ x) (in x3 slot)
	# now that we've used z to compute fz, dfz, use the z memory slot
	# as a buffer (z will no longer hold h ⊛ x)
    R = CartesianIndices(axes(hk))
    for (ax,i) in zip(R,1:length(hk))
        # ax loops over the axes of each element in h
        # e.g., ax might be CartesianIndex(-1,0), for which ax.I is (-1,0)
        # i loops over the vectorized index, used for building dh

		# sb.x1 = dfz .* circshift(x,-1*ax.I)
        circshift!(sb.x1, x, -1 .*ax.I); hadamard!(sb.x1,dfz)

		# sb.x1 = ̃h ⊛ (dfz .* circshift(x,-1*ax.I))
        padxj!(sb.xpad1, z, sb.padval)
        conv!(sb.x1, sb.xpad1, γ.hrev[k])

        v = @view dxdh[:,i];
		plus!(v, v, sb.x1) # dxdh[:,i] += h ⊛ (dfz .* circshift(x,-1*ax.I))
		circshift!(sb.x1, fz, ax.I); # sb.x1 = (f.(z))^(ax) = f.(z^(ax))
		plus!(v, v, sb.x1) # += f.(z^(ax))
    end
	lmul!(exp(γ.β[0]+γ.β[k]),dxdh)
    return dxdh
end

"""
Take the derivative w.r.t. βk of (∇10𝚽(x;γ)=̂x-y+ e^β0 ∑_k e^βk Ck' dφ.(Ck x))
    which is simply e^(β0+βk) Ck' dφ.(Ck x).
"""
function _d2Φdxdβ!(dβ,x,γ,𝚽,k,sb)
    # give memory slots easier to understand names
	xpad = sb.xpad2; padxj!(xpad, x, sb.padval)
    z = sb.x2 # use the x2 memory slot to hold hk ⊛ x
    fz = sb.x1 # use the x1 memory slot to hold dφ.(hk ⊛ x)
    fzpad = sb.xpad1 # use this memory slot to hold padded fz

    # get the derivative
    conv!(z,xpad,γ.h[k]) # z = hk ⊛ x
    copy!(fz, z); 𝚽.reg.dφ(fz) # reg uses in-place operations, so result is now in fz
    padxj!(fzpad,fz,sb.padval)
    conv!(dβ,fzpad,γ.hrev[k]) # res = Ck'fz = Ck' dφ.(Ck x)
	lmul!(exp(γ.β[0]+γ.β[k]), dβ) # dβ .*= exp(γ.β[0]+γ.β[k])
    return dβ
end


# TODO - this assumes that A'A is the Hessian of the data-fit term!
"""
∇20Φ = _grad20!(res, v, x, γ, 𝚽, sb)
res = ∇20Φ(x) * v
"""
function _grad20!(res, v, x, γ, 𝚽, sb)
	v = view(v,:); v = reshape(v,axes(x))

	z = sb.x3 # will use this to store hk ⊛ x
    vtemp = sb.x2

	xpad = sb.xpad1
	padxj!(xpad, x, sb.padval)

    fill!(res, 0)
    for k = 1:γ.K # loop over all filters
        conv!(z, xpad, γ.h[k])         # z = hk ⊛ x (stored in sb.x3)
        padxj!(sb.xpad2, v, sb.padval) # overwrites sb.xpad2 to be vpad
		conv!(vtemp, sb.xpad2, γ.h[k]) # vtemp = Ck*v
        𝚽.reg.ddφ(z)      			   # z is updated in-place as ddφ(hk ⊛ x)
		hadamard!(vtemp, z) 		   # vtemp = diag(ddφ(hk ⊛ x)) * (Ck*v)

        padxj!(sb.xpad2, vtemp, sb.padval) # renews sb.xpad2 as vpad
        conv!(vtemp, sb.xpad2, γ.hrev[k])  # vtemp = Ck'diag(ddφ(hk ⊛ x))(Ck*v)
		lmul!(exp(γ.β[k]), vtemp)		  # vtemp = e^βk Ck'diag(ddφ(hk⊛x))*(Ck*v)
        plus!(res, res, vtemp)
    end
    # res = e^β0 ∑_k e^βk Ck' diag(ddφ(hk ⊛ x)) (Ck*v)
	lmul!(exp(γ.β[0]), res)

	# result should be res + A'Av
	tmp_y = @view sb.y1[:]; mul!(tmp_y, 𝚽.A, vec(v)) 		# sb.y1 = A*vec(v)
	tmp_x = @view sb.x3[:]; mul!(tmp_x, 𝚽.At, tmp_y)  	# sb.x3 = A'A*vec(v)
	# res = A'Av + e^β0 ∑_k e^βk Ck'diag(ddφ(hk ⊛ x))Ck*v
	plus!(res, res, sb.x3)
    return res
end

# TODO - this currently modifies CG_init if it's the same size as xjhat
# it's helpful but maybe unexpected behavoir?
function ift_gradient(xjhat, xjtrue, yj, γ, 𝚽, ℓ;
	sb=scratch(xjhat, γ.h; y=yj), sb2=scratch(xjhat, γ.h; y=yj),
	CG_N=size(𝚽.A,2), CG_init=[],
	dh = deepcopy(γ.h), dβ = deepcopy(γ.β))

	# H*x computes the Hessian times a vector
	# H*y = res = A'Ay + ∑_k e^βk Ck' * diag(ddφ(hk ⊛ x)) * (Ck*y) = ∇_{x,x} Φ * v
	H = LinearMap((res,v) -> _grad20!(res,v, xjhat, γ, 𝚽, sb2),
					length(xjhat), length(xjhat), issymmetric=true, isposdef=true)
	# TODO it could be positive *semi* definite... depending on A
	# but theory requires PD - maybe warn the user?

	# approximate H^-1 * (̂x-xtrue)
	er = @view view(sb.xpad1, axes(xjhat)...)[:]; minus!(er, xjhat, xjtrue)
	# get the initialization for CG which will also hold the result from CG
	if isempty(CG_init)
		cgres = @view copy(xjhat)[:]
		copyto!(cgres, er) # equivalent to approximating Hessian as Identity
	elseif length(CG_init)==length(xjhat)
		cgres = @view CG_init[:]
	elseif isa(CG_init,Number)
		cgres = @view copy(xjhat)[:]
		fill!(cgres,CG_init) # 0 is default for cg!()
	else # user supplied CG_init of incorrect size
		@warn("Initialization for CG must be a float or of size(xjhat)."*
			"Invalid size "*size(CG_init)*" provided. Initializing to 0.")
		cgres = @view copy(xjhat)[:]
		fill!(cgres,0) # 0 is default for cg!()
	end
	fill!(sb.x1,0); #fill!(sb.x2,0); fill!(sb.x3, 0)
	# vec is non-allocating
	sv = CGStateVariables(vec(sb.x1), vec(sb.x2), vec(sb.x3))
	cg!(cgres, H, er; maxiter=CG_N, statevars = sv)

	for k=1:γ.K; fill!(dh[k],0); end
	fill!(dβ, 0)
	dxdβ = zeros(size(xjhat))
	dxdh = zeros(length(xjhat),length(γ.h[1]))
	for k = 1:γ.K # update gradient for each filter
         if length(γ.h[k]) != size(dxdh,2) # TODO - not tested
			 dxdh = zeros(length(xjhat),length(γ.h[k]))
		 else
			 fill!(dxdh, 0)
		 end
        _d2Φdxdh!(dxdh,xjhat,γ,𝚽,k,axes(xjhat),sb)
        _d2Φdxdβ!(dxdβ,xjhat,γ,𝚽,k,sb)

		dh[k] = - setoffsets(reshape(dxdh'*cgres,size(dh[k])))
		dβ[k] = - vec(dxdβ)' * cgres
    end
	return dh, dβ, cgres
end

mutable struct BilevelState
	# input variables (generally required from user)
	xtrue # Array of clean training samples, length L
	y # Array of noisy training samples, length L
	γ::Hyperparameters # hyperparameters
	niter # maximum number of upper-level iterations (U in paper)

	# input variables that come with reasonable defaults
	ℓ::LossFunction # upper-level loss function
	𝚽::CostFunction # lower-level cost function
	fun::Function # user output function
	descend_h::Bool
	descend_β::Bool

	# defining the lower-level optimizer
	lloptimizer

	# defining the gradient computation and upper-level functions
	dℓdγ
	isconverged
	# TODO add a checkconvergence field for user to define their own function
	updateγ!

	# iternal variables - momentum coefficients and such
	iter::Int # # of completed upper-level iterations
	γold::Hyperparameters
	xhat # array of length l for all current reconstructed images

	# other internal variables
	fcost
	sb # [vector of] ScratchBuffers
	L # number of training signals
	dh
	dβ

	# dictionary for holding method-specific options
	opt::Dict

	# user output
	out
end

# γ, niter, ℓ, 𝚽

function updateγ_ADAM!(γ, opth, optβ, dh, dβ; descend_h=false, descend_β=false)
	if descend_h
		[Flux.Optimise.update!(opth[k], γ.h[k], dh[k]) for k=1:γ.K]
	end
	if descend_β
		dβ[0] = 0 # don't descend with repect to β0, TODO: make optional
		Flux.Optimise.update!(optβ, γ.β, dβ)
	end
end
function updateγ_gd!(γ, αℓ, dh, dβ; descend_h=false, descend_β=false)
	if descend_h
		for k=1:γ.K; minus!(γ.h[k], γ.h[k], αℓ*dh[k]); end;
	end
	if descend_β
		dβ[0] = 0 # don't descend with repect to β0, TODO: make optional
		minus!(γ.β, γ.β, αℓ*dβ)
	end
end
function _lloptimizer!(s::BilevelState, l::Int; α𝚽=[])
	copy!(s.opt["gmopt"][l].x0, s.xhat[l])
	s.opt["gmopt"][l].f_grad! = (res, x; pogm_state) -> f_grad!(res, x, s.y[l],
		s.γ, s.𝚽, pogm_state, s.sb[l])
	s.opt["gmopt"][l].f_L = s.𝚽.LC + regLC(s.γ,s.𝚽.reg)
	reinit_pogm_state!(s.opt["gmopt"][l]; momchanged=false)
	if !isempty(α𝚽) # since re-init will overwrite this setting
		s.opt["gmopt"][l].alpha = α𝚽
	end
	pogm_restart!(s.opt["gmopt"][l])
	return s.opt["gmopt"][l]
end

# defaults to ADAM for upper-level
function initializebilevelstate(xtrue, y, γ;
	xhat=length(xtrue[1])==length(y[1]) ? deepcopy(y) : fill!(deepcopy(xtrue),0),
	#xhat = [reshape(𝚽.At*vec(y[l]),size(xtrue[l])) for l=1:length(y)],
	niter=100, # upper level iterations
	T = 100, # lower level iterations
	ℓ=initℓ(),
	reg=reg_2norm!(),
	𝚽=init𝚽(xtrue[1]; reg, opt=:denoising),
	sb=[scratch(xtrue[l], γ.h; y=y[l]) for l=1:length(xtrue)],
	sb2=[scratch(xtrue[l], γ.h; y=y[l]) for l=1:length(xtrue)],
	opt = Dict{String,Any}("h"=>[Flux.ADAM() for k=1:length(γ.h)], "β"=>Flux.ADAM(),
			"cgres" => [zeros(size(xl)) for xl in xtrue],
			"gmopt"=>[gm_Φ(y[l],γ,𝚽; x0=xhat[l], niter=T, # alpha=α𝚽,
					mom=:pogm, checkconvergence=s->_checkconvergence_gnorm(s; gnorm=1e-10))
					for l=1:length(xtrue)]),
	lloptimizer = _lloptimizer!,
	dℓdγ = (s::BilevelState, l) -> ift_gradient(s.xhat[l],xtrue[l],y[l],s.γ,s.𝚽,s.ℓ;
		sb=s.sb[l], sb2=sb2[l], CG_init=s.opt["cgres"][l]),
	fun=(s,l,llres)->bileveluserfun(s,l,llres; printfreq=100),
	descend_h=true, descend_β=true,
	iter=0,
	updateγ! = (s) -> updateγ_ADAM!(s.γ, s.opt["h"], s.opt["β"], s.dh, s.dβ;
							descend_h, descend_β),
	isconverged = (s) -> (s.iter >= s.niter),
    out = similar(Array{Any}, niter+1) # pre-allocate output of fun evals
	)

	L = length(xtrue)
    @assert L == length(y) "Must input the same number of clean and noisy signals"
	@assert descend_h | descend_β "Must descend on h and/or β"

	γold = initγ(; h=deepcopy(γ.h), β=copy(γ.β), α=γ.α, B=γ.B)

	@assert all([size(y[l]) == size(y[1]) for l=2:L]) "support for different"*
		" input sizes not yet implemented"
	# theory is easy, but need to properly change or index sb
	# to extend to the case of different sized input signals

	convflag = false
	fcost = NaN
	dh = []
	dβ = []

	s = BilevelState(xtrue, y, γ,
		niter,
		ℓ, 𝚽,
		fun,
		descend_h, descend_β,
		lloptimizer,
		dℓdγ, isconverged, updateγ!,
		iter, γold, xhat, fcost, sb,
		L, dh, dβ,
		opt, out)

	s.fun(s,0,[]) # call user output before taking the first iteration
	return s
end

# specific for AID_BIO method
# to match the paper settings, user must provide α𝚽 > 0
# else it will change each upper-level iteration
function initializebilevelstate_aidbio(xtrue, y, γ, αℓ, T;
	α𝚽=[], Ncg=length(xtrue[1]),
	xhat = length(xtrue[1])==length(y[1]) ? deepcopy(y) : fill!(deepcopy(xtrue),0),
	 kwargs...)

	return initializebilevelstate(xtrue, y, γ;
		lloptimizer=(s::BilevelState, l::Int) -> #_lloptimizer!(s,l;α𝚽),
		 	denoise_caol!(gm_Φ(y[l],s.γ,s.𝚽;
					x0=s.xhat[l], niter=T, sb=s.sb[l], fun=gm_user_fun, alpha=α𝚽,
					mom=:pgm, restart=:none))[end],
		dℓdγ = (s::BilevelState, l) -> ift_gradient(s.xhat[l],xtrue[l],y[l],s.γ,
			s.𝚽,s.ℓ; sb=s.sb[l], CG_N=Ncg, CG_init=s.opt["cgres"][l]),
		fun=function (s,l, llres)
				bileveluserfun(s,l, llres; printfreq=100)
				if s.iter == 0 && l>0 && l <= s.L
					copy!(s.opt["cgres"][l], s.xhat[l])
					# first initialization for CG is xhat
					# after that, will use previous CG result
				end
			end,
		updateγ! = (s) -> updateγ_gd!(s.γ, αℓ, s.dh, s.dβ;
			descend_h=s.descend_h, descend_β=s.descend_β),
		opt = Dict{String,Any}(
			"cgres" => [zeros(size(xtrue[l])) for l=1:length(xtrue)],
			"gmopt"=>[gm_Φ(y[l],γ,𝚽;
					x0=xhat[l], niter=T, fun=gm_user_fun, alpha=α𝚽, mom=:pogm,
					checkconvergence=s->_checkconvergence_gnorm(s; gnorm=1e-10))
					for l=1:length(xtrue)]),
		values(kwargs)...)
end

function initializebilevelstate_ttsa(xtrue, y, γ, αℓ;
	α𝚽=[], 𝚽=init𝚽(xtrue[1]; reg, opt=:denoising),
	x0 = [denoise_caol(y[l],γ,𝚽; niter=Int(10e6),
		checkconvergence=s->_checkconvergence_gnorm(s; gnorm=1e-10))[1] for l=1:length(y)],
	Ncg=size(𝚽.A,2), kwargs...)

	return initializebilevelstate(xtrue, y, γ; xhat=x0,
		lloptimizer=(s::BilevelState, l::Int) -> denoise_caol!(gm_Φ(y[l],s.γ,s.𝚽;
					x0=s.xhat[l], niter=1, # only one LL step each outer-loop
					sb=s.sb[l], fun=gm_user_fun, alpha=α𝚽))[end],
					# TODO - can save memory by passing xnew = xhat[l]
		dℓdγ = (s::BilevelState, l) -> ift_gradient(s.xhat[l],xtrue[l],y[l],s.γ,
			s.𝚽,s.ℓ; sb=s.sb[l], CG_N=Ncg, CG_init=s.opt["cgres"][l]),
		fun=(s,l,llres)->bileveluserfun(s,l,llres; printfreq=100),
		updateγ! = (s) -> updateγ_gd!(s.γ, αℓ, s.dh, s.dβ;
			descend_h=s.descend_h, descend_β=s.descend_β),
		opt = Dict{String,Any}("cgres"=>[zeros(size(xl)) for xl in xtrue]),
		values(kwargs)...)
end

function bileveluserfun(s::BilevelState, l, llres; printfreq=100)
	if mod(s.iter,printfreq)==0
	end
	if l > s.L && s.iter >= 0
		s.out[s.iter+1] = (s.iter, s.fcost)
	end
end


function bilevel_learn_H!(s::BilevelState)
    β1end = @view s.γ.β[1:end] # helpful for gradient

	lk = ReentrantLock()

    while !s.isconverged(s) && s.iter < s.niter # outer iterations
		# prepare for the iteration
		[copy!(s.γold.h[k], s.γ.h[k]) for k=1:s.γ.K]
		copy!(s.γold.β, s.γ.β)

        # recreate these variables each loop, since K could change
        s.dh = deepcopy(s.γ.h); s.dh .*= 0 # fill! doesn't work for vectors of vectors
		s.dβ = deepcopy(s.γ.β); fill!(s.dβ,0)

		# compute/estimate the upper-level gradient
		s.fcost = 0

		Threads.@threads for l = 1:s.L
			#@show Threads.threadid()
			if size(s.y[1]) != size(s.y[l]) # TODO - not yet implemented
            end

			# lower-level optimization
			llres = s.lloptimizer(s, l)
			copy!(s.xhat[l], llres.xnew)

			lock(lk) do # call user function
				s.fun(s, l, llres)
			end
		end

		for l = 1:s.L # these are not (currently) thread-safe
			# evaluate upper-level and get upper-level gradient
			s.fcost += s.ℓ.loss(s.xhat[l],s.xtrue[l],s.γ)/s.L
			dhl, dβl = s.dℓdγ(s, l)
			plus!(s.dh, s.dh, dhl)
			plus!(s.dβ, s.dβ, dβl)
        end

        # take a gradient step in the upper-level parameters
		s.updateγ!(s)
		validateγ!(s.γ)

		# update output variable for the user
		s.iter = s.iter + 1 # tracks the number of iterations *completed*
        s.fun(s, s.L+1,[])
    end
	validateγ!(s.γ)
	return s.γ, s.out, s
end
