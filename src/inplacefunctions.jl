using LinearAlgebra
using Random: seed!, randperm

include("regularizers.jl") # defines common regularizer functions

sosdiff(a::Number,b::Number) = abs2(a-b)
sosdiff(A,B) = sum(ab -> sosdiff(ab...),zip(A,B)) # quick version of norm(vec(A-B))^2
function minus!(res, a1, a2)
	for i in eachindex(res)
      res[i] = a1[i] - a2[i]
   end
end
function plus!(res, a1, a2)
	for i in eachindex(res)
      res[i] = a1[i] + a2[i]
   end
end

function hadamard!(A,B)
    @assert size(A) == size(B)
    for i in eachindex(A)
         @inbounds A[i] *= B[i]
    end
    return A
end
