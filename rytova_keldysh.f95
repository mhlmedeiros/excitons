module Rytova_Keldysh

    implicit none
    real, parameter :: EPSILON_0    = 55.26349406 ! e^2 GeV^(-1)fm^(-1) == e^2 (1e9 eV 1e-15 m)^(-1)
    real, parameter :: PI           = 4.*atan(1.0)
    real, parameter :: Vkk_const    = -1.e6/(2. * EPSILON_0 * (2*PI)**2)
contains
    subroutine potential_average(k_x, k_y, delta_k, n_sub, eps, r_0, mean_pot)
        implicit none
        ! INPUT VARIABLES
        real, intent(in)    :: k_x, k_y, delta_k
        real, intent(in)    :: eps, r_0
        integer, intent(in) :: n_sub
        ! OUTPUT VARIABLES
        real, intent(out)   :: mean_pot
        ! LOCAL VARIABLES
        ! real, external      :: V_Rytova_Keldysh
        real                :: kx_ini, ky_ini
        real                :: kx_sweep, ky_sweep, dk_sub, k_norm
        real                :: v_sum
        integer             :: i, j, N_total

        ! PREPARATION
        dk_sub      = delta_k/(n_sub - 1)
        kx_ini      = k_x - delta_k/2.
        ky_ini      = k_y - delta_k/2.
        N_total     = n_sub**2
        v_sum       = 0.

        do i = 0, n_sub - 1
            kx_sweep = kx_ini + real(i)*dk_sub
            do j = 0, n_sub - 1
                ky_sweep = ky_ini + real(j)*dk_sub
                k_norm   = SQRT(kx_sweep**2 + ky_sweep**2)
                if ( k_norm /= 0 ) then
                    v_sum = v_sum + 1.0/(eps*k_norm + r_0*k_norm**2)
                else
                    N_total = N_total - 1
                end if
            end do
        end do
        mean_pot = v_sum/real(N_total)
        !
    end subroutine potential_average

    subroutine build_potential_matrix(kx_flat, ky_flat, N_total, N_sub, eps, r_0, V)
        implicit none
        ! INPUT VARIABLES
        integer, intent(in) :: N_total, N_sub
        real, intent(in)    :: kx_flat(N_total), ky_flat(N_total)
        real, intent(in)    :: eps
        real, intent(in)    :: r_0
        ! OUTPUT VARIABLES
        real, intent(out)   :: V(N_total,N_total)
        ! LOCAL VARIABLES
        real, parameter     :: PI = 4 * atan(1.0)
        real                :: dk, kx, ky
        integer             :: i, j, N_mesh


        ! PREPARATION
        N_mesh = int(sqrt(real(N_total)))
        dk = kx_flat(2) - kx_flat(1)
        V = 0.

        ! MAIN LOOP (AVERAGES)
        do i = 1, N_mesh
            do j = i+1, N_total
                kx = kx_flat(i) - kx_flat(j)
                ky = ky_flat(i) - ky_flat(j)
                call potential_average(kx, ky, dk, N_sub, eps, r_0, V(i,j))
            end do
        end do

        ! COPY AND PASTE THE VALUES IN THE RIGHT PLACE
        do i = 1, N_mesh-1
            V(i*N_mesh+1: (i+1)*N_mesh, i*N_mesh+1:) = V(:N_mesh,:N_total-i*N_mesh)
        end do

        ! SUMMATION WITH THE TRANSPOSED
        V = V + transpose(V)

        ! DIAGONAL VALUE (IT IS THE SAME FOR ALL ELEMENTS: k_x=0, ky=0)
        do i=1, N_total
            call potential_average(0., 0., dk, N_sub, eps, r_0, V(i,i))
        end do
        V = Vkk_const * (dk**2) * V
        !
    end subroutine build_potential_matrix

end module Rytova_Keldysh

! function V_Rytova_Keldysh(k, eps, r_0)
!     !   Purpose:
!     !   To return the value assumed for the
!     !   Ritova-Keldysh (constant aside)
!     !
!     implicit none
!
!     ! Data dictionary: Declarations, types & units
!     real, intent(in) :: k       ! Length of the k vector        [nm^-1]
!     real, intent(in) :: eps     ! Effective dielectric constant [no units]
!     real, intent(in) :: r_0     ! The screening length          [nm]
!     real :: V_RK
!     V_RK = 1.0/(eps*k + r_0*k**2)
! end function V_RK
