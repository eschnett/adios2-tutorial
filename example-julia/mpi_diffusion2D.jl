import MPI
using Plots

comm = MPI.COMM_WORLD

# MPI
MPI.Init()
nprocs = MPI.Comm_size(comm)
dims = [0,0]
MPI.Dims_create!(nprocs, dims)
comm_cart   = MPI.Cart_create(comm, dims, [0,0], 1) #
me          = MPI.Comm_rank(comm_cart)              #
coords      = MPI.Cart_coords(comm_cart)
neighbors_x = MPI.Cart_shift(comm_cart, 0, 1)
neighbors_y = MPI.Cart_shift(comm_cart, 1, 1)

@views function update_halo(A, neighbour_x, neighbour_y)
    # left neighboour x dim
    if neighbour_x[1] != MPI.MPI_PROC_NULL
        sendbuf = A[2,:]
        recvbuf = zeros(size(A[1,:]))
        MPI.Send(sendbuf, neighbour_x[1], 0, comm)
        MPI.Recv!(recvbuf, neighbour_x[1], 1, comm)
        A[1,:] = recvbuf
    end
    
    # right neighbour x dim
    if neighbour_x[2] != MPI.MPI_PROC_NULL
        sendbuf = A[end-1,:]
        recvbuf = zeros(size(A[end,:]))
        MPI.Send(sendbuf,  neighbors_x[2], 1, comm)
        MPI.Recv!(recvbuf, neighbors_x[2], 0, comm)
        A[end,:] = recvbuf
    end 

    # top neighbour y dim
    if neighbors_y[1] != MPI.MPI_PROC_NULL
        sendbuf = A[:,2]
        recvbuf = zeros(size(A[:,1]))
        MPI.Send(sendbuf,  neighbors_y[1], 2, comm)
        MPI.Recv!(recvbuf, neighbors_y[1], 3, comm)
        A[:,1] = recvbuf
    end
    
    # bottom neighbour y
    if neighbors_y[2] != MPI.MPI_PROC_NULL
        sendbuf = A[:,end-1]
        recvbuf = zeros(size(A[:,end]))
        MPI.Send(sendbuf,  neighbors_y[2], 3, comm)
        MPI.Recv!(recvbuf, neighbors_y[2], 2, comm)
        A[:,end] = recvbuf
    end
    return
end

@views function diffusion2D()
    # Physics
    lam        = 1.0                 # Thermal conductivity
    cp_min     = 1.0                 # Minimal heat capacity
    lx, ly     = 10.0, 10.0          # Length of computational domain in dimension x and y

    # Numerics
    nx, ny     = 128, 128            # Number of gridpoints in dimensions x and y
    nt         = 20000               # Number of time steps
    nx_g       = dims[1]*(nx-2) + 2  # Number of gridpoints of the global problem in dimension x
    ny_g       = dims[2]*(ny-2) + 2  # ...                                        in dimension y
    dx         = lx/(nx_g-1)         # Space step in dimension x
    dy         = ly/(ny_g-1)         # ...        in dimension y

    # Array initializations
    T     = zeros((nx,   ny, ))
    Cp    = zeros((nx,   ny, ))
    dTedt = zeros((nx-2, ny-2))
    qTx   = zeros((nx-1, ny-2))
    qTy   = zeros((nx-2, ny-1))

    # Initial conditions (heat capacity and temperature with two Gaussian anomalies each)
    x0    = coords[1]*(nx-2)*dx
    y0    = coords[2]*(ny-2)*dy

    Cp .= cp_min .+ reshape([  5*exp(-((x0 + ix*dx - lx/1.5)/1.0)^2 - ((y0 + iy*dy - ly/1.5)/1.0)^2) + 
                                5*exp(-((x0 + ix*dx - lx/1.5)/1.0)^2 - ((y0 + iy*dy - ly/3.0)/1.0)^2) for ix=1:nx for iy=1:ny], (nx,ny))

    T  .=          reshape([100*exp(-((x0 + ix*dx - lx/3.0)/2.0)^2 - ((y0 + iy*dy - ly/2.0)/2.0)^2) +
                                50*exp(-((x0 + ix*dx - lx/1.5)/2.0)^2 - ((y0 + iy*dy - ly/2.0)/2.0)^2) for ix=1:nx for iy=1:ny], (nx,ny))



    # Time loop
    nsteps = 50                                                      # Number of times data is written during the simulation
    dt     = (min(dx,dy)^2)*cp_min/lam/4.1                           # Time step for the 2D Heat diffusion
    t      = 0                                                       # Initialize physical time

    # setting up Animation
    gr()
    ENV["GKSwstype"]="nul"
    anim = Animation();
    nx_v = (nx-2)*dims[1];
    ny_v = (ny-2)*dims[2];
    T_v  = zeros(nx_v, ny_v);
    T_nohalo = zeros(nx-2, ny-2);

    tic = time()                                                                            # Start for measuring time elapsed
    tic = time()
    for x in 1:nt
        if x % nsteps == 0                                                                  # Visualize only every nstepth time step
            T_nohalo .= T[2:end-1,2:end-1];                                                 # Copy data removing the halo.
            if (me==0) heatmap(transpose(T_nohalo), aspect_ratio=1); frame(anim); end       # Visualize it on rank 0.
        end
        qTx       .= -lam*diff(T[:,2:end-1], dims=1)/dx                                     # Fourier's law of heat conduction: q_x   = -λ δT/δx
        qTy       .= -lam*diff(T[2:end-1,:], dims=2)/dy                                     # ...                               q_y   = -λ δT/δy
        dTedt     .= 1.0 ./ Cp[2:end-1,2:end-1] .* (-diff(qTx, dims=1)/dx .-                # Conservation of energy:           δT/δt = 1/cₚ(-δq_x/δx
                                                    diff(qTy, dims=2)/dy)                   #                                               - δq_y/dy)
        T[2:end-1,2:end-1] = T[2:end-1,2:end-1] + dt*dTedt                                  # Update of temperature             T_new = T_old + δT/δt
        t         = t + dt                                                                  # Elapsed physical time
        update_halo(T, neighbors_x, neighbors_y)
    end
    tac = time()                                                                            # Stop for measuring time elapsed
    println("Time elapsed ", tac - tic, " sec")                                             # Print Time elapsed
    println("Min. temperature ", minimum(T))                                                # Min temperature
    println("Max. temperature ", maximum(T))                                                # Max temperature
    if (me==0) mp4(anim, "diffusion2D.mp4", fps = 15) end                                   # Save Animation in .mp4
    # if (me==0) gif(anim, "diffusion2D.gif", fps = 15) end                                   # Create a .gif file
end

diffusion2D()
