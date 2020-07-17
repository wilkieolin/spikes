using ProgressMeter, LinearAlgebra, Distributions, PyPlot
import DSP.conv

function default_params()
    params = Dict{String,Any}()
    #network parameters
    params["n_signals"] = 2
    params["n_neurons"] = 20

    #learning parameters
    params["dt"] = 1e-3
    params["eps_rc"] = 1e-3
    params["eps_ff"] = 1e-4
    params["scale_ff"] = 0.18
    params["scale_rc"] = 1.0/0.9
    params["quad_cost"] = 0.02/0.9

    #cell parameters
    params["membrane_leak"] = 50.0
    params["threshold"] = 0.5

    #data parameters
    params["data_amplitude"] = 2000
    params["data_smoothing"] = 30

    return params
end

function compute_leak_constant(params)
    return (1 - params["membrane_leak"]*params["dt"])
end

function initialize_ff(params)
    n_signals = params["n_signals"]
    n_neurons = params["n_neurons"]
    
    d = Normal()
    wgts = 0.5*rand(d, n_signals, n_neurons)
    wgts = wgts ./ (sqrt.(ones(n_signals, 1) * sum(wgts.^2, dims=1)))
    return wgts
end

function initialize_rc(params)
    n_neurons = params["n_neurons"]
    threshold = params["threshold"]
    
    wgts = -0.2 .* rand(n_neurons, n_neurons) .- threshold .* Matrix(I, n_neurons, n_neurons)
    return wgts
end

function generate_data(time, params)
    return generate_data(time, params["n_signals"], params["data_smoothing"], params["data_amplitude"])
end

function generate_data(time::Int, n_signals::Int, sigma::Real, amplitude::Real)
    k_length = 1000
    k_mid = Int(k_length / 2)
    
    d = Normal()
    
    kernel = (1 / (sigma * sqrt(2*pi))) .* exp.(-((collect(1:k_length) .- k_mid).^2) ./ (2*sigma^2))
    kernel = kernel ./ sum(kernel)
    
    signal = rand(d, n_signals, time)
    for i in 1:n_signals
        convolved = amplitude .* conv(signal[i,:], kernel)
        signal[i,:] = convolved[k_mid:time+k_mid-1]
    end
    
    return signal
end

function run_simple(signal::Array{<:Real,2}, fw::Array{<:Real,2}, rc::Array{<:Real,2}, params::Dict{String,Any})
    #parameters
    n_signals, n_time = size(signal)
    n_neurons = params["n_neurons"]
    dt = params["dt"]
    threshold = params["threshold"]
    eps_ff = params["eps_ff"]
    eps_ff = params["eps_rc"]
    quad_cost = params["quad_cost"]
    scale_ff = params["scale_ff"]
    scale_rc = params["scale_rc"]
    
    #integration/run variables
    voltage = zeros(Float64, n_neurons, n_time)
    spikes = zeros(Float64, n_neurons, n_time)
    rates = zeros(Float64, n_neurons, n_time)
    ind = CartesianIndex(1)
    
    #constants
    v_decay = (1 - params["membrane_leak"]*dt)
    d = Normal()
    
    for i in 2:n_time
        leakage = v_decay .* voltage[:,i-1]
        forward = dt .* fw' * signal[:,i-1]
        recurrent = rc * spikes[:,i-1]
        noise = 0.001*rand(d, n_neurons)

        voltage[:,i] = leakage .+ forward .+ recurrent .+ noise
        
        (maxval, ind) = findmax(voltage[:,i] .- threshold .- 0.01*rand(d, n_neurons))
        
        if (maxval >= 0.0)
            spikes[ind,i] = 1.0
        end
        
        rates[:,i] = v_decay .* rates[:,i-1] .+ spikes[:,i]
    end
    
    return voltage, spikes, rates
end

function learn_simple(signal, fw, rc, params)
    #parameters
    n_signals, n_time = size(signal)
    n_neurons = params["n_neurons"]
    dt = params["dt"]
    threshold = params["threshold"]
    eps_ff = params["eps_ff"]
    eps_rc = params["eps_rc"]
    quad_cost = params["quad_cost"]
    scale_ff = params["scale_ff"]
    scale_rc = params["scale_rc"]
    
    #weight samples
    log_periods = floor(Int,log(2, n_time))
    fw_samples = zeros(Float64, log_periods, n_signals, n_neurons)
    rc_samples = zeros(Float64, log_periods, n_neurons, n_neurons)
    
    #integration/run variables
    filtered_signal = zeros(Float64, n_signals)
    voltage = zeros(Float64, n_neurons)
    rates = zeros(Float64, n_neurons)
    spk = Float64(0.0)
    ind = CartesianIndex(1)
    exp_interval = Int64(1)
    
    #constants
    neuron_identity = Matrix(I,n_neurons,n_neurons)
    v_decay = (1 - params["membrane_leak"]*dt)
    d = Normal()
    
    @showprogress for i in 2:n_time
        #register the weights on a logarithmic interval
        if (mod(i, 2^exp_interval) == 0)
            fw_samples[exp_interval,:,:] = fw
            rc_samples[exp_interval,:,:] = rc
            exp_interval += 1
        end
        leakage = v_decay .* voltage
        forward = dt .* fw' * signal[:,i]
        recurrent = spk .* rc[:,ind]
        noise = 0.001*rand(d,n_neurons)

        voltage =  leakage .+ forward .+ recurrent .+ noise
        filtered_signal = v_decay .* filtered_signal .+ dt .* signal[:,i]
        
        (maxval, ind) = findmax(voltage .- threshold .- 0.01*rand(d, n_neurons))
        
        if (maxval >= 0.0)
            spk = Float64(1.0)
            
            #learn on the feed-forward connections
            fw[:,ind] .= fw[:,ind] .+ eps_ff .* (scale_ff .* filtered_signal .- fw[:,ind])
            #learn on recurrent connections
            scaled_voltage = scale_rc .* (voltage .+ quad_cost .* rates)
            rc[:,ind] .= rc[:,ind] .- eps_rc .* (scaled_voltage .+ rc[:,ind] .+ quad_cost .* neuron_identity[:,ind])
            #update the firing rates after learning is done
            rates[ind] += 1.0
        else
            spk = Float64(0.0)
        end
        
        rates = v_decay .* rates
    end
    
    return fw, rc, fw_samples, rc_samples
end

function leaky_integrate(signal, params)
    n_signals, n_time = size(signal)
    
    leaky_signal = zeros(Float64, n_signals, n_time)
    leak_constant = compute_leak_constant(params)
    
    for i in 2:n_time
        leaky_signal[:,i] = leak_constant.*leaky_signal[:,i-1] .+ signal[:,i-1]
    end
    
    return leaky_signal
end

function compute_optimal_decoder(signal, fw, rc, params)
    n_signals = params["n_signals"]
    n_neurons = params["n_neurons"]
    
    leaky_signal = leaky_integrate(signal, params)
    v,s,r = run_simple(signal, fw, rc, params)
    decoder = pinv(r)' * leaky_signal'
    
    return decoder
end

function compute_optimal_decoders(fw_samples, rc_samples, params)
    n_samples, n_signals, n_neurons = size(fw_samples)
    
    decoders = zeros(Float64, n_samples, n_neurons, n_signals)
    for i in 1:n_samples
        signal = generate_data(50000, n_signals, params["data_smoothing"], 0.3*params["data_amplitude"])
        leaky_signal = leaky_integrate(signal, params)
        fw = fw_samples[i,:,:]
        rc = rc_samples[i,:,:]
        
        v,s,r = run_simple(signal, fw, rc, params)
        decoder = pinv(r)' * leaky_signal'
        decoders[i,:,:] = decoder
    end
    
    return decoders
end

function reconstruct(signal, fw, rc, decoder, params)
    leaky_signal = leaky_integrate(signal, params)
    
    _,_,rates = run_simple(signal, fw, rc, params)
    signal_est = rates' * decoder
    
    return leaky_signal, signal_est'
end

function compute_learning_metrics(fw_samples::Array{<:Real,3}, rc_samples::Array{<:Real,3}, decoders::Array{<:Real,3}, params::Dict{String,Any}; n_trials::Int=10, n_time::Int=10000, take_mean::Bool=true)
    n_samples, n_signals, n_neurons = size(fw_samples)
    
    mean_rates = zeros(Float64, n_samples, n_trials)
    errors = zeros(Float64, n_samples, n_trials)
    membrane_variance = zeros(Float64, n_samples, n_trials)
    
    for i in 1:n_trials
        #generate a signal and its LI form
        signal = generate_data(n_time, params)
        leaky_signal = leaky_integrate(signal, params)
        
        for j in 1:n_samples
            fw = fw_samples[j,:,:]
            rc = rc_samples[j,:,:]
            decoder = decoders[j,:,:]
            
            #get the estimate for that network
            v,s,r = run_simple(signal, fw, rc, params)
            est_signal = r' * decoder
            
            #compute the metrics
            errors[j,i] = sum(var(est_signal' .- leaky_signal, dims=2)) / sum(var(leaky_signal, dims=2))
            mean_rates[j,i] = sum(s) / (params["dt"]*n_neurons*n_time)
            membrane_variance[j,i] = sum(var(v,dims=2)) / n_neurons
        end
    end
    
    if take_mean
        avg = x->mean(x, dims=2)
        return map(avg, (errors, mean_rates, membrane_variance))
    else
        return errors, mean_rates, membrane_variance
    end
end

