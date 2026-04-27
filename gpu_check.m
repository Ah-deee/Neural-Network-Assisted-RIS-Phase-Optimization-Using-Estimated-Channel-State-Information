function useGPU = gpu_check()
%GPU_CHECK Detect if a compatible GPU is available for computation.
%   useGPU = GPU_CHECK() returns true if a CUDA-capable GPU is available
%   and Parallel Computing Toolbox is present; otherwise it returns false.
%
%   This helper is used to configure the ExecutionEnvironment of
%   Deep Learning Toolbox training and prediction.

useGPU = false;

try
    g = gpuDevice();
    if ~isempty(g)
        useGPU = true;
    end
catch
    useGPU = false;
end

end

