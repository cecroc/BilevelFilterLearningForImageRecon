# define regularizer! structure
# assumed that all the regularizer functions operate in-place!
struct Regularizer! # R(x; γ) in our set-up above
    φ
    dφ  # derivative of φ function
    ddφ # second derivative of φ function
    Ldφ # maximum value of ddφ for use in computing Lipschitz constant
    mystr # info string that you can print to know about the regularizer
end

# φ = corner rounded 1-norm:  φ(z) = √( |z|^2 + ϵ^2 )
reg_cr1n! = (ϵ)->Regularizer!(z->map!(z->sqrt(abs2(z)+ϵ^2),z,z),
                             z->map!(z->z/sqrt(abs2(z)+ϵ^2),z,z),
                             z->map!(z->ϵ^2/(abs2(z)+ϵ^2)^(3/2),z,z),
                             1/ϵ, "cr1n "*string(ϵ))
# φ = geman and mcclure potential  φ(z) = z^2 / (ϵ + z^2)
reg_gm! = (ϵ) -> Regularizer!(z->map!(z->z^2/(abs2(z)+ϵ),z,z),
                            z->map!(z->2*z*ϵ/(abs2(z)+ϵ)^2,z,z),
                            z->map!(z->2*ϵ*(ϵ-3*z^2)/(abs2(z)+ϵ)^3,z,z),
                            2/ϵ, "gm "*string(ϵ))
# welsh potential  φ(z) = 1 - exp(-z^2)
reg_welsh! = (ϵ) -> Regularizer!(z->map!(z->1-exp(-ϵ*abs2(z)),z,z),
                               z->map!(z->2*ϵ*z*exp(-ϵ*abs2(z)),z,z),
                               z->map!(z->-2*ϵ*exp(-ϵ*abs2(z))*(2*ϵ*abs2(z)-1),z,z),
                               2*ϵ, "welsh "*string(ϵ))
# 2-norm potential  φ(z) = 1/2 |z|^2
reg_2norm! = ( ) -> Regularizer!(z->map!(z->1/2*abs2(z),z,z),
                             z->map!(z->z,z,z),
                             z->map!(z->1,z,z),
                             1, "2 norm ")
