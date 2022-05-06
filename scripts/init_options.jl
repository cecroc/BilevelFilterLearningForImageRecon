using FFTW: dct
using Distributions
using Random: randperm, rand

# these are functions more helpful in the user scripts
function getDCTfilters(;filterdims=(3,3), dc=false)
	if length(filterdims) != 2
		@error "haven't implemented getDCTfilters for non-2d filterdims"
	end
	temp = dct(Matrix(I,filterdims),1)
	h = kron(temp,temp)' / sqrt(prod(filterdims)) # 2d dct filters
	h = [setoffsets(reshape(h[:,i],filterdims)) for i=1:prod(filterdims)]
	if !dc # drop the DC filter
		h = h[2:end] # drop the first (constant) filter
	end
	# can visualize h with:
	# using MIRTjim: jim
	# jim(cat(h...,dims=3)); title!("DCT filters")
	return h
end

"""
function to generate a random piece-wise constant signal
"""
function makestep(;dim=32, njump=3, valueDist=Normal(0,1), minsep=2)
    jump_locations = _jumplocations(dim=dim, njump=njump, minsep=minsep)
    index = zeros(dim)
    index[jump_locations] .= 1
    index = cumsum(index)
    values = rand(valueDist, njump) # random signal values
    x = zeros(dim)
    for jj=1:njump
        x[index .== jj] .= values[jj]
    end
    x[index .== 0] .= values[njump] # periodic end conditions
    x = circshift(x, rand(1:dim, 1)) # random shift - just to be sure
    return x
end

"""
function to generate a random piece-wise constant signal
"""
function _jumplocations(;dim=32, njump=3, minsep=2)
    if dim < 100
        jump_locations = randperm(dim)[1:njump]
        while minimum(diff([1; jump_locations])) <= minsep
            jump_locations = randperm(dim)[1:njump] # random jump locations
        end
    else
        jump_locations = []
        n = min(100,dim) # number of indices to figure out at once
        nseg = Integer(ceil(dim/n)) # how many times we'll execute the for loop
        for i = 1:nseg
            jl = sort(randperm(n)[1:Integer(ceil(njump/nseg))])
            while minimum(diff([1; jl])) <= minsep
                jl = sort(randperm(n)[1:Integer(ceil(njump/nseg))]) # random jump locations
            end
            jump_locations = [jump_locations; (i-1)*100 .+ jl]
        end
        jump_locations = jump_locations[1:njump] # in case dim % 100 != 0
    end
    return jump_locations
end
