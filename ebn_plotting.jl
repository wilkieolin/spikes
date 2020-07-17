using PyPlot
include("ebn_julia.jl")

function plot_reconstruction(fw_samples, rc_samples, decoders, index, params; n_time::Int=1000)
    signal = generate_data(n_time, params)
    leaky_signal = leaky_integrate(signal, params)
    
    _,_,rates = run_simple(signal, fw_samples[index,:,:], rc_samples[index,:,:], params)
    signal_est = rates' * decoders[index,:,:]
    
    plot(signal_est, color="orange")
    plot(leaky_signal', color="blue")
end