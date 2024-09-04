template <typename T>
class ThreadSafeVector
{
public:
    __host__ ThreadSafeVector() : size_(0), capacity_(100)
    {
        // Allocate memory for the vector and size on the device
        hipMalloc(&data_, capacity_ * sizeof(T));
        hipMalloc(&d_size_, sizeof(int));

        // Initialize size on the device
        hipMemcpy(d_size_, &size_, sizeof(int), hipMemcpyHostToDevice);
    }
    __host__ ThreadSafeVector(size_t capacity) : size_(0), capacity_(capacity)
    {
        // Allocate memory for the vector and size on the device
        hipMalloc(&data_, capacity * sizeof(T));
        hipMalloc(&d_size_, sizeof(int));

        // Initialize size on the device
        hipMemcpy(d_size_, &size_, sizeof(int), hipMemcpyHostToDevice);
    }

    __host__ __device__ ~ThreadSafeVector()
    {
        if (!hipGetLastError())
        {
            hipFree(data_);
            hipFree(d_size_);
        }
    }

    __device__ void append(const T &value)
    {
        // Get the current index using atomic operation
        int index = atomicAdd(d_size_, 1);

        // Check if the index is within bounds
        if (index < capacity_)
        {
            data_[index] = value;
        }
        else
        {
            // Roll back the size in case of overflow
            atomicSub(d_size_, 1);
        }
    }

    __host__ int getSize() const
    {
        int host_size;
        hipMemcpy(&host_size, d_size_, sizeof(int), hipMemcpyDeviceToHost);
        return host_size;
    }

    __host__ __device__ T *data() const
    {
        return data_;
    }
    __host__ __device__ int *size() const
    {
        return d_size_;
    }
    __host__ __device__ T &operator[](int index)
    {
        return data_[index];
    }

private:
    T *data_;         // Pointer to the data on the device
    int *d_size_;     // Pointer to the size on the device
    int size_;        // Size variable on the host (for initialization)
    size_t capacity_; // Capacity of the vector
};
