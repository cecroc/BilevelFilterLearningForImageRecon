using LinearAlgebra
using Random: randn

# can do res = A*x or mul!(res,A,x) for in-place, efficient version!
# assume x comes in vectorized form
# res should be in vectorized form as well
# (reshape as needed in calling functions)
function _scalingA!(y,x,scalefactor)
	copy!(y,x)
	y .*= scalefactor
end
function inpaintingAt!(x,y,mask)
	fill!(x,0) # missing entries are coded as 0
	xmask = @view x[mask]
	copy!(xmask, y)
end
function defineA(;xdim=[],ydim=xdim,opt=:denoising,
    sampfrac=0.9,
    scalefactor=1.)

    if opt == :I
        A = I; At = I
    elseif opt == :denoising
        @assert !(isempty(xdim)) "xdim must be set for denoising option"
        A = LinearMap(copy!, prod(xdim), prod(xdim),
            issymmetric=true, isposdef=true, ismutating=true)
        At = A
		La = 1.
    elseif opt == :scaled
        @assert !(isempty(xdim)) "xdim must be set for scaled option"
        A = LinearMap((y,x) -> _scalingA!(y,x,scalefactor),
			prod(xdim), prod(xdim),
            issymmetric=true, isposdef=(scalefactor>0), ismutating=true)
        At = A
		La = abs(scalefactor)^2
    elseif opt == :inpainting
        # do a random sampling pattern
		mask = rand(xdim...) .< sampfrac
		mask = vec(mask)
		A = LinearMap((y,x) -> copy!(y, x[mask]),sum(mask), prod(xdim))
		At = LinearMap((x,y) -> inpaintingAt!(x,y,mask), prod(xdim), sum(mask))
		La = 1.
    elseif opt == :random
        A = scalefactor.*randn(prod(ydim),prod(xdim))
        At = A'
		La = opnorm(A)^2
    else
        @warn "Unknown option "*string(opt)*" for the system matrix. "*
            "Defaulting to A=I."
        A = I; At = I; La = 1.
    end
    return A, At, La
end
