! wrfcol.F90 RUNDIR : WRF's aer_opt = 1 chain on four test columns, with WRF's own routines (wrfaer.F90):
! aerosol_in's read and horizontal interpolation (its loop body, verbatim but for one point), aer_time_int,
! aer_p_int, and rrtmg_sw's iaer = 6 combination (its loop, verbatim).  Prints, per column and layer, the six
! aerod values and the 14 RRTMG-band tau / ssa / asy, as text the test oracle is cut from.
program wrfcol
  use wrfaer
  implicit none
  integer, parameter :: alevsiz = 12, no_months = 12, no_src = 6, latsiz = 46, lonsiz = 72
  integer, parameter :: nlay = 30, ncolt = 4
  real :: pina(alevsiz), lat_od(latsiz), lon_od(lonsiz)
  real :: aerodin(lonsiz, alevsiz, latsiz, no_months, no_src)
  real :: aerodm(1, alevsiz, 1, no_months, no_src), aerodt(1, alevsiz, 1, no_src)
  real :: p(1, 1:nlay + 1, 1), pf(1, 1:nlay + 1, 1), aerod(1, 1:nlay + 1, 1, no_src), totaod(1, 1)
  real :: xlat(ncolt), xlong(ncolt), scale(ncolt), phalf(nlay + 1)
  real(kind=8) :: ecaer(nlay, 6), ztaua, zomga, zasya
  character(len=256) :: dir
  integer :: i, k, m, ks, j, c, ib, ia, ilon1, ilon2, jlat1, jlat2, julday
  real :: wlat1, wlat2, wlon1, wlon2, julian

  call get_command_argument(1, dir)
  open(27, file=trim(dir)//'/aerosol_plev.formatted', form='formatted', status='old')
  do k = 1, alevsiz
    read (27, *) pina(k)
  end do
  close(27)
  open(28, file=trim(dir)//'/aerosol_lat.formatted', form='formatted', status='old')
  do j = 1, latsiz
    read (28, *) lat_od(j)
  end do
  close(28)
  open(29, file=trim(dir)//'/aerosol_lon.formatted', form='formatted', status='old')
  do j = 1, lonsiz
    read (29, *) lon_od(j)
  end do
  close(29)
  open(30, file=trim(dir)//'/aerosol.formatted', form='formatted', status='old')
  do ks = 1, no_src
  do m = 1, no_months
  do j = 1, latsiz
  do k = 1, alevsiz
  do i = 1, lonsiz
    read (30, *) aerodin(i, k, j, m, ks)
  end do
  end do
  end do
  end do
  end do
  close(30)
  call swaerpr

  ! The test columns: Sahara, open Pacific, N Mexico, beside the date line (WRF longitudes).
  xlat = (/ 22.0, 2.0, 26.0, -30.0 /)
  xlong = (/ 10.0, -150.0, -105.0, 179.9 /)
  scale = (/ 0.97, 1.0, 0.85, 1.0 /)
  ! 1976-03-20 12Z: day of year 80 (leap year), WRF julian 79.5.
  julday = 80
  julian = 79.5
  do k = 1, nlay + 1
    phalf(k) = 101325.0 + (100.0 - 101325.0) * real(k - 1) / real(nlay)
  end do
  do c = 1, ncolt
    ! aerosol_in's horizontal step for this one point
    call interp_vec(lat_od, xlat(c), .true., jlat1, jlat2, wlat1, wlat2)
    call interp_vec(lon_od, xlong(c), .true., ilon1, ilon2, wlon1, wlon2)
    do ks = 1, no_src
    do m = 1, no_months
    do k = 1, alevsiz
      aerodm(1, k, 1, m, ks) = wlon1 * (wlat1 * aerodin(ilon1, k, jlat1, m, ks) + &
                                        wlat2 * aerodin(ilon1, k, jlat2, m, ks)) + &
                               wlon2 * (wlat1 * aerodin(ilon2, k, jlat1, m, ks) + &
                                        wlat2 * aerodin(ilon2, k, jlat2, m, ks))
    end do
    end do
    end do
    call aer_time_int(julday, julian, aerodm, aerodt, alevsiz, no_months, no_src, &
                      1, 1, 1, 1, 1, nlay + 1, 1, 1, 1, 1, 1, nlay + 1, 1, 1, 1, 1, 1, nlay)
    ! full (interface) and half (layer) pressures, bottom-up, the model's: interfaces scaled, the top one at 100 Pa
    do k = 1, nlay + 1
      pf(1, k, 1) = phalf(k) * scale(c)
    end do
    pf(1, nlay + 1, 1) = 100.0
    do k = 1, nlay
      p(1, k, 1) = 0.5 * (pf(1, k, 1) + pf(1, k + 1, 1))
    end do
    call aer_p_int(p, pina, alevsiz, aerodt, aerod, no_src, pf, totaod, &
                   1, 1, 1, 1, 1, nlay + 1, 1, 1, 1, 1, 1, nlay + 1, 1, 1, 1, 1, 1, nlay)
    do ia = 1, 6
      do k = 1, nlay
        ecaer(k, ia) = aerod(1, k, 1, ia)
      end do
    end do
    write (*, '(a,i2,a,f8.3,a,f9.3,a,es16.8)') 'column ', c, ' lat ', xlat(c), ' lon ', xlong(c), ' totaod ', totaod(1, 1)
    do k = 1, nlay
      write (*, '(a,i3,2es18.10,6es18.10)') 'layer', k, p(1, k, 1), pf(1, k, 1) - pf(1, k + 1, 1), (aerod(1, k, 1, ia), ia = 1, 6)
      do ib = 1, 14
        ztaua = 0.d0; zasya = 0.d0; zomga = 0.d0
        do ia = 1, 6
          ztaua = ztaua + rsrtaua(ib, ia) * ecaer(k, ia)
          zomga = zomga + rsrtaua(ib, ia) * ecaer(k, ia) * rsrpiza(ib, ia)
          zasya = zasya + rsrtaua(ib, ia) * ecaer(k, ia) * rsrpiza(ib, ia) * rsrasya(ib, ia)
        end do
        if (zomga /= 0.d0) zasya = zasya / zomga
        if (ztaua /= 0.d0) zomga = zomga / ztaua
        write (*, '(a,i3,3es24.16)') 'band', ib, ztaua, zomga, zasya
      end do
    end do
  end do
end program wrfcol
