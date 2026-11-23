! WRF v4.6.1 routines, verbatim: interp_vec (phys/module_physics_init.F), aer_time_int and aer_p_int
! (phys/module_radiation_driver.F), swaerpr's tables (phys/module_ra_rrtmg_sw.F).  Public domain (WRF) /
! AER BSD-3 (swaerpr).  Only wrf_error_fatal is stubbed.
module wrfaer
implicit none
real(kind=8) :: rsrtaua(14,6), rsrpiza(14,6), rsrasya(14,6)
contains
subroutine wrf_error_fatal(m)
  character(len=*) :: m
  print *, m; stop 1
end subroutine
  subroutine interp_vec(locvec,locwant,periodic,loc1,loc2,wght1,wght2)

  implicit none

  real, intent(in), dimension(:) :: locvec
  real, intent(in)               :: locwant
  logical, intent(in)            :: periodic
  integer, intent(out)           :: loc1, loc2
  real, intent(out)              :: wght1, wght2

  integer :: vsize, n
  real    :: locv1, locv2

  vsize = size(locvec)

  loc1 = -1
  loc2 = -1

  do n = 1, vsize-1
    if ( locvec(n) <= locwant .and. locvec(n+1) > locwant ) then
      loc1  = n
      loc2  = n+1
      locv1 = locvec(n)
      locv2 = locvec(n+1)
      exit
    end if
  end do

  if ( loc1 < 0 .and. loc2 < 0 ) then
    if ( periodic ) then
      if ( locwant < locvec(1) ) then
        loc1  = vsize
        loc2  = 1
        locv1 = locvec(vsize)-360.0
        locv2 = locvec(1)
      else
        loc1  = vsize
        loc2  = 1
        locv1 = locvec(vsize)
        locv2 = locvec(1)+360.0
      end if
    else
      if ( locwant < locvec(1) ) then
        loc1  = 1
        loc2  = 1
        locv1 = locvec(1)
        locv2 = locvec(1)
      else
        loc1  = vsize
        loc2  = vsize
        locv1 = locvec(vsize)
        locv2 = locvec(vsize)
      end if
    end if
  end if

  wght2 = (locwant-locv1) / (locv2-locv1)
  wght1 = 1.0 - wght2

  return
  end subroutine interp_vec
SUBROUTINE aer_time_int(julday,julian,aerodm,aerodt,levsiz,num_months,no_src,  &
                              ids , ide , jds , jde , kds , kde ,     &
                              ims , ime , jms , jme , kms , kme ,     &
                              its , ite , jts , jte , kts , kte )

! adapted from oznint from CAM module
!  input: aerodm - read from physics_init
! output: aerodt - time interpolated

!  USE module_ra_cam_support, ONLY : getfactors

   IMPLICIT NONE

   INTEGER,    INTENT(IN) ::           ids,ide, jds,jde, kds,kde, &
                                       ims,ime, jms,jme, kms,kme, &
                                       its,ite, jts,jte, kts,kte

   INTEGER,      INTENT(IN   )    ::   levsiz, num_months, no_src

   REAL,  DIMENSION( ims:ime, levsiz, jms:jme, num_months, no_src ),      &
          INTENT(IN   ) ::                                  aerodm

   INTEGER, INTENT(IN )      ::        JULDAY
   REAL,    INTENT(IN )      ::        JULIAN

   REAL,  DIMENSION( ims:ime, levsiz, jms:jme, no_src ),      &
          INTENT(OUT  ) ::                                  aerodt

   !Local
   REAL      :: intJULIAN
   integer   :: np1,np,nm,m,k,i,j,s
   integer   :: IJUL
   integer, dimension(12) ::  date_oz
   data date_oz/16, 45, 75, 105, 136, 166, 197, 228, 258, 289, 319, 350/
   real, parameter :: daysperyear = 365.  ! number of days in a year
   real      :: cdayozp, cdayozm
   real      :: fact1, fact2, deltat
   logical   :: finddate
   logical   :: ozncyc
   CHARACTER(LEN=256) :: msgstr

   ozncyc = .true.
   ! JULIAN starts from 0.0 at 0Z on 1 Jan.
   intJULIAN = JULIAN + 1.0       ! offset by one day
! jan 1st 00z is julian=1.0 here
   IJUL=INT(intJULIAN)
!  Note that following will drift.
!    Need to use actual month/day info to compute julian.
   intJULIAN=intJULIAN-FLOAT(IJUL)
   IJUL=MOD(IJUL,365)
   IF(IJUL.EQ.0)IJUL=365
   intJULIAN=intJULIAN+IJUL
   np1=1
   finddate=.false.

!  do m=1,num_months
   do m=1,12
      if(date_oz(m).gt.intjulian.and..not.finddate) then
        np1=m
        finddate=.true.
      endif
   enddo
   cdayozp=date_oz(np1)

   if(np1.gt.1) then
      cdayozm=date_oz(np1-1)
      np=np1
      nm=np-1
   else
      cdayozm=date_oz(12)
      np=np1
      nm=12
   endif

!  call getfactors(ozncyc,np1, cdayozm, cdayozp,intjulian, &
!                   fact1, fact2)
!
! Determine time interpolation factors.  Account for December-January
! interpolation if dataset is being cycled yearly.
!
   if (ozncyc .and. np1 == 1) then                      ! Dec-Jan interpolation
      deltat = cdayozp + daysperyear - cdayozm
      if (intjulian > cdayozp) then                     ! We are in December
         fact1 = (cdayozp + daysperyear - intjulian)/deltat
         fact2 = (intjulian - cdayozm)/deltat
      else                                              ! We are in January
         fact1 = (cdayozp - intjulian)/deltat
         fact2 = (intjulian + daysperyear - cdayozm)/deltat
      end if
   else
      deltat = cdayozp - cdayozm
      fact1 = (cdayozp - intjulian)/deltat
      fact2 = (intjulian - cdayozm)/deltat
   end if
!
! Time interpolation.
!
      do s=1, no_src
      do j=jts,jte
      do k=1,levsiz
      do i=its,ite
            aerodt(i,k,j,s) = aerodm(i,k,j,nm,s)*fact1 + aerodm(i,k,j,np,s)*fact2
      end do
      end do
      end do
      end do

END SUBROUTINE aer_time_int

SUBROUTINE aer_p_int(p ,pin, levsiz, aerodt, aerod, no_src, pf, totaod,   &
                     ids , ide , jds , jde , kds , kde ,     &
                     ims , ime , jms , jme , kms , kme ,     &
                     its , ite , jts , jte , kts , kte )

!-----------------------------------------------------------------------
!
! Purpose: Interpolate aerosol from current time-interpolated values to model levels
!
! Method: Use pressure values to determine interpolation levels
!
! Author: Bruce Briegleb
! WW: Adapted for general use
!
!   p:  model level pressure at half levels (Pa, bottom-up)
!   pf: model level pressure at full levles (Pa, bottom-up)
!
!--------------------------------------------------------------------------
   implicit none
!--------------------------------------------------------------------------
!
! Arguments
!
   INTEGER,    INTENT(IN) ::           ids,ide, jds,jde, kds,kde, &
                                       ims,ime, jms,jme, kms,kme, &
                                       its,ite, jts,jte, kts,kte

   integer, intent(in) :: levsiz              ! number of aerosol layers
   integer, intent(in) :: no_src              ! types of aerosol 

   real, intent(in) :: p(ims:ime,kms:kme,jms:jme)
   real, intent(in) :: pf(ims:ime,kms:kme,jms:jme)
   real, intent(in) :: pin(levsiz)        ! aerosol data level pressures (mks, top-down)
   real, intent(in) :: aerodt(ims:ime,levsiz,jms:jme,1:no_src) ! aerosol optical depth

   real, intent(out) :: aerod(ims:ime,kms:kme,jms:jme,1:no_src) ! aerosol optical depth
   real, intent(out) :: totaod(ims:ime,jms:jme)                 ! total aerosol optical depth
!
! local storage
!
   real    pmid(its:ite,kts:kte)
   integer i,j                 ! longitude index
   integer k, kk, kkstart, kout! level indices
   integer kupper(its:ite)     ! Level indices for interpolation
   integer kount               ! Counter
   integer ncol, pver, s

   real    dpu                 ! upper level pressure difference
   real    dpl                 ! lower level pressure difference
   real    dpm                 ! pressure difference in a model layer surrounding half p

   ncol = ite - its + 1
   pver = kte - kts + 1

   do s=1,no_src
   do j=jts,jte
!
! Initialize index array
!
   do i=its, ite
      kupper(i) = 1
   end do
!
! The pressure from incoming data is in hPa and top-down, 
!     while model pressure is in Pa and bottom-up
!
      do k = kts,kte
         kk = kte - k + kts
      do i = its,ite
         pmid(i,kk) = p(i,k,j)*0.01
      enddo
      enddo

   do k=1,pver

      kout = pver - k + 1
!
! Top level we need to start looking is the top level for the previous k
! for all longitude points
!
      kkstart = levsiz
      do i=its,ite
         kkstart = min0(kkstart,kupper(i))
      end do
      kount = 0
!
! Store level indices for interpolation
!
      do kk=kkstart,levsiz-1
         do i=its,ite
            if (pin(kk).lt.pmid(i,k) .and. pmid(i,k).le.pin(kk+1)) then
               kupper(i) = kk
               kount = kount + 1
            end if
         end do
!
! If all indices for this level have been found, do the interpolation and
! go to the next level
!
         if (kount.eq.ncol) then
            do i=its,ite
               dpu = pmid(i,k) - pin(kupper(i))
               dpl = pin(kupper(i)+1) - pmid(i,k)
               dpm = pf(i,kout,j) - pf(i,kout+1,j)
               aerod(i,kout,j,s) = (aerodt(i,kupper(i),j,s)*dpl + &
                             aerodt(i,kupper(i)+1,j,s)*dpu)/(dpl + dpu)
               aerod(i,kout,j,s) = aerod(i,kout,j,s)*dpm
            end do
            goto 35
         end if
      end do
!
! If we've fallen through the kk=1,levsiz-1 loop, we cannot interpolate and
! must extrapolate from the bottom or top aerosol data level for at least some
! of the longitude points.
!
      do i=its,ite
         if (pmid(i,k) .lt. pin(1)) then
            dpm = pf(i,kout,j) - pf(i,kout+1,j)
            aerod(i,kout,j,s) = aerodt(i,1,j,s)*pmid(i,k)/pin(1)
            aerod(i,kout,j,s) = aerod(i,kout,j,s)*dpm
         else if (pmid(i,k) .gt. pin(levsiz)) then
            dpm = pf(i,kout,j) - pf(i,kout+1,j)
            aerod(i,kout,j,s) = aerodt(i,levsiz,j,s)
            aerod(i,kout,j,s) = aerod(i,kout,j,s)*dpm
         else
            dpu = pmid(i,k) - pin(kupper(i))
            dpl = pin(kupper(i)+1) - pmid(i,k)
            dpm = pf(i,kout,j) - pf(i,kout+1,j)
            aerod(i,kout,j,s) = (aerodt(i,kupper(i),j,s)*dpl + &
                          aerodt(i,kupper(i)+1,j,s)*dpu)/(dpl + dpu)
            aerod(i,kout,j,s) = aerod(i,kout,j,s)*dpm
         end if
      end do

      if (kount.gt.ncol) then
         call wrf_error_fatal ('AER_P_INT: Bad aerosol data: non-monotonicity suspected')
      end if
35    continue

   end do
   end do
   end do

   do j=jts,jte
   do i=its,ite
      totaod(i,j) = 0.
   end do
   end do

   do s=1,no_src
   do j=jts,jte
   do k=1,pver
   do i=its,ite
      totaod(i,j) = totaod(i,j) + aerod(i,k,j,s)
   end do
   end do
   end do
   end do

   return
END SUBROUTINE aer_p_int
subroutine swaerpr
      rsrtaua( 1, :) = (/ &
        0.10849_8, 0.66699_8, 0.65255_8, 0.11600_8, 0.06529_8, 0.04468_8/)
      rsrtaua( 2, :) = (/ &
        0.10849_8, 0.66699_8, 0.65255_8, 0.11600_8, 0.06529_8, 0.04468_8/)
      rsrtaua( 3, :) = (/ &
        0.20543_8, 0.84642_8, 0.84958_8, 0.21673_8, 0.28270_8, 0.10915_8/)
      rsrtaua( 4, :) = (/ &
        0.20543_8, 0.84642_8, 0.84958_8, 0.21673_8, 0.28270_8, 0.10915_8/)
      rsrtaua( 5, :) = (/ &
        0.20543_8, 0.84642_8, 0.84958_8, 0.21673_8, 0.28270_8, 0.10915_8/)
      rsrtaua( 6, :) = (/ &
        0.20543_8, 0.84642_8, 0.84958_8, 0.21673_8, 0.28270_8, 0.10915_8/)
      rsrtaua( 7, :) = (/ &
        0.20543_8, 0.84642_8, 0.84958_8, 0.21673_8, 0.28270_8, 0.10915_8/)
      rsrtaua( 8, :) = (/ &
        0.52838_8, 0.93285_8, 0.93449_8, 0.53078_8, 0.67148_8, 0.46608_8/)
      rsrtaua( 9, :) = (/ &
        0.52838_8, 0.93285_8, 0.93449_8, 0.53078_8, 0.67148_8, 0.46608_8/)
      rsrtaua(10, :) = (/ &
        1.69446_8, 1.11855_8, 1.09212_8, 1.72145_8, 1.03858_8, 1.12044_8/)
      rsrtaua(11, :) = (/ &
        1.69446_8, 1.11855_8, 1.09212_8, 1.72145_8, 1.03858_8, 1.12044_8/)
      rsrtaua(12, :) = (/ &
        1.69446_8, 1.11855_8, 1.09212_8, 1.72145_8, 1.03858_8, 1.12044_8/)
      rsrtaua(13, :) = (/ &
        1.69446_8, 1.11855_8, 1.09212_8, 1.72145_8, 1.03858_8, 1.12044_8/)
      rsrtaua(14, :) = (/ &
        0.10849_8, 0.66699_8, 0.65255_8, 0.11600_8, 0.06529_8, 0.04468_8/)
 
      rsrpiza( 1, :) = (/ &
        .5230504_8, .7868518_8, .8531531_8, .4048149_8, .8748231_8, .2355667_8/)
      rsrpiza( 2, :) = (/ &
        .5230504_8, .7868518_8, .8531531_8, .4048149_8, .8748231_8, .2355667_8/)
      rsrpiza( 3, :) = (/ &
        .8287144_8, .9949396_8, .9279543_8, .6765051_8, .9467578_8, .9955938_8/)
      rsrpiza( 4, :) = (/ &
        .8287144_8, .9949396_8, .9279543_8, .6765051_8, .9467578_8, .9955938_8/)
      rsrpiza( 5, :) = (/ &
        .8287144_8, .9949396_8, .9279543_8, .6765051_8, .9467578_8, .9955938_8/)
      rsrpiza( 6, :) = (/ &
        .8287144_8, .9949396_8, .9279543_8, .6765051_8, .9467578_8, .9955938_8/)
      rsrpiza( 7, :) = (/ &
        .8287144_8, .9949396_8, .9279543_8, .6765051_8, .9467578_8, .9955938_8/)
      rsrpiza( 8, :) = (/ &
        .8970131_8, .9984940_8, .9245594_8, .7768385_8, .9532763_8, .9999999_8/)
      rsrpiza( 9, :) = (/ &
        .8970131_8, .9984940_8, .9245594_8, .7768385_8, .9532763_8, .9999999_8/)
      rsrpiza(10, :) = (/ &
        .9148907_8, .9956173_8, .7504584_8, .8131335_8, .9401905_8, .9999999_8/)
      rsrpiza(11, :) = (/ &
        .9148907_8, .9956173_8, .7504584_8, .8131335_8, .9401905_8, .9999999_8/)
      rsrpiza(12, :) = (/ &
        .9148907_8, .9956173_8, .7504584_8, .8131335_8, .9401905_8, .9999999_8/)
      rsrpiza(13, :) = (/ &
        .9148907_8, .9956173_8, .7504584_8, .8131335_8, .9401905_8, .9999999_8/)
      rsrpiza(14, :) = (/ &
        .5230504_8, .7868518_8, .8531531_8, .4048149_8, .8748231_8, .2355667_8/)

      rsrasya( 1, :) = (/ &
        0.700610_8, 0.818871_8, 0.702399_8, 0.689886_8, .4629866_8, .1907639_8/)
      rsrasya( 2, :) = (/ &
        0.700610_8, 0.818871_8, 0.702399_8, 0.689886_8, .4629866_8, .1907639_8/)
      rsrasya( 3, :) = (/ &
        0.636342_8, 0.802467_8, 0.691305_8, 0.627497_8, .6105750_8, .4760794_8/)
      rsrasya( 4, :) = (/ &
        0.636342_8, 0.802467_8, 0.691305_8, 0.627497_8, .6105750_8, .4760794_8/)
      rsrasya( 5, :) = (/ &
        0.636342_8, 0.802467_8, 0.691305_8, 0.627497_8, .6105750_8, .4760794_8/)
      rsrasya( 6, :) = (/ &
        0.636342_8, 0.802467_8, 0.691305_8, 0.627497_8, .6105750_8, .4760794_8/)
      rsrasya( 7, :) = (/ &
        0.636342_8, 0.802467_8, 0.691305_8, 0.627497_8, .6105750_8, .4760794_8/)
      rsrasya( 8, :) = (/ &
        0.668431_8, 0.788530_8, 0.698682_8, 0.657422_8, .6735182_8, .6519706_8/)
      rsrasya( 9, :) = (/ &
        0.668431_8, 0.788530_8, 0.698682_8, 0.657422_8, .6735182_8, .6519706_8/)
      rsrasya(10, :) = (/ &
        0.729019_8, 0.803129_8, 0.784592_8, 0.712208_8, .7008249_8, .7270548_8/)
      rsrasya(11, :) = (/ &
        0.729019_8, 0.803129_8, 0.784592_8, 0.712208_8, .7008249_8, .7270548_8/)
      rsrasya(12, :) = (/ &
        0.729019_8, 0.803129_8, 0.784592_8, 0.712208_8, .7008249_8, .7270548_8/)
      rsrasya(13, :) = (/ &
        0.729019_8, 0.803129_8, 0.784592_8, 0.712208_8, .7008249_8, .7270548_8/)
      rsrasya(14, :) = (/ &
        0.700610_8, 0.818871_8, 0.702399_8, 0.689886_8, .4629866_8, .1907639_8/)
end subroutine swaerpr
end module wrfaer
