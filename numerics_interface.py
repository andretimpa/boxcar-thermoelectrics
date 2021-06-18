#########################
#       LIBRARIES       #
#########################

import numpy as np
import numpy.polynomial as poly
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema as extrema
from math import inf as INF
from scipy.special import spence as dilog
try:
    from scipy.optimize import toms748 as bracket
except ImportError:
    from scipy.optimize import brentq as bracket
import warnings
from matplotlib.widgets import TextBox, Button

np.seterr(all='warn')
#warnings.filterwarnings('error')

#global constants

pi2_6 = np.pi**2/6      # RiemannZeta(2)

###############################
#       BASIC FUNCTIONS       #
###############################

#the basic functions for calculating f, g, Delta and their relevant derivatives


#Fermi occupation
#
# beta = Inverse Temperature
# mu   = Chemical Potential
# eps  = Energy of the Electron
def F(beta, mu, eps):
    return 1/(1+np.exp(beta*(eps-mu)))

#in the following functions the arguments are as follows:
#
# bR  = Inverse Temperature in the Right Lead
# bL  = Inverse Temperature in the Left Lead
# mR  = Chemical Potential in the Right Lead
# mL  = Chemical Potential in the Left Lead
# eps = Energy of the Electron

#Delta (difference of Fermi occupations)
def Delta(bR, bL, mR, mL, eps):
    ER = np.exp(bR*(eps-mR))
    EL = np.exp(bL*(eps-mL))
    return (EL - ER)/((1+ER)*(1+EL))

#g/Delta (the function that we must cross with a line to get the boxcars)
def g_D(bR, bL, mR, mL, eps):
    ER = np.exp(bR*(eps-mR))
    EL = np.exp(bL*(eps-mL))
    return (ER*(EL+1)**2 + EL*(ER+1)**2)/((EL-ER)*(EL+1)*(ER+1))

#Delta/g
def D_g(bR, bL, mR, mL, eps):
    ER = np.exp(bR*(eps-mR))
    EL = np.exp(bL*(eps-mL))
    return ((EL-ER)*(EL+1)*(ER+1))/(ER*(EL+1)**2 + EL*(ER+1)**2)

#first derivative of (g/Delta)
def dg_D(bR, bL, mR, mL, eps):
    ER = np.exp(bR*(eps-mR))
    EL = np.exp(bL*(eps-mL))
    fermiL = (1+EL)**2
    fermiR = (1+ER)**2
    L_R    = EL**2 + ER**2
    return (bR*fermiL*ER*(L_R + 2*EL*(1+ER+ER**2)) - bL*fermiR*EL*(L_R + 2*ER*(1+EL+EL**2)))/((EL-ER)**2*fermiL*fermiR)

#first derivative of (Delta/g)
def dD_g(bR, bL, mR, mL, eps):
    ER = np.exp(bR*(eps-mR))
    EL = np.exp(bL*(eps-mL))
    fermiL = (1+EL)**2
    fermiR = (1+ER)**2
    L_R    = EL**2 + ER**2
    return (bL*fermiR*EL*(L_R + 2*ER*(1+EL+EL**2)) - bR*fermiL*ER*(L_R + 2*EL*(1+ER+ER**2)))/(EL*fermiR + ER*fermiL)**2

#Regularized second derivative of g/Delta: (g/Delta)''(Delta/g)^3
#This function has the same roots as the second derivative, however it is not singular
def d2g_DR(bR, bL, mR, mL, eps):
    ER = np.exp(bR*(eps-mR))
    EL = np.exp(bL*(eps-mL))
    fermiL = (1+EL)**3
    fermiR = (1+ER)**3
    boseL  = EL-1
    boseR  = ER-1
    DE3    = ER**3 - EL**3
    ELR    = EL*ER
    gE     = EL + ER + ELR*(EL + ER + 4)
    return (bL**2*EL*fermiR*(ELR*(2+EL*(9+EL*(3+2*EL))) + (2+EL*(3+EL*(9+2*EL)))*ER**2 -boseL*DE3) + bR**2*fermiL*ER*(boseR*DE3 + \
ELR*(2+ER*(9+ER*(3+2*ER))) + EL**2*(2+ER*(3+ER*(9+2*ER)))) -4*bL*bR*ELR*fermiL*fermiR*(EL+ER))/(gE**3)

#################################
#      AUXILIARY FUNCTIONS      #
#################################

#in the following functions the arguments are as follows:
#
# bR   = Inverse Temperature in the Right Lead
# bL   = Inverse Temperature in the Left Lead
# mR   = Chemical Potential in the Right Lead
# mL   = Chemical Potential in the Left Lead
# prec = Precision (Absolute Error Tolerance)

#Calculate the energy range that we need to consider in order to make the calculations with the desired precision
# THIS SHOULD NOT BE CALLED DIRECTLY, ONLY THROUGH get_structure
def get_region(bR, bL, mR, mL, prec):
    prec  = np.abs(prec)
    logR  = np.log(bR*prec)
    logL  = np.log(bL*prec)
    eminR = mR + logR/bR
    eminL = mL + logL/bL
    emaxR = mR - logR/bR
    emaxL = mL - logL/bL
    energ = max([np.abs(x) for x in [eminR, eminL, emaxR, emaxL]])
    prec /= energ
    logR  = np.log(bR*prec)
    logL  = np.log(bL*prec)
    eminR = mR + logR/bR
    eminL = mL + logL/bL
    emaxR = mR - logR/bR
    emaxL = mL - logL/bL
    return min([eminR, eminL]), max([emaxR, emaxL])

#Returns the energy range, together with the inflxion points of g/Delta in that interval (calculated by brute force)
def get_structure(bR, bL, mR, mL, prec, force_region = None):
    #####
    if force_region is None:
        if bR == bL and mR == mL:
            return None
        emin, emax = get_region(bR, bL, mR, mL, prec)
        while emin > emax:
            prec /= 10
            emin, emax = get_region(bR, bL, mR, mL, prec)
    # We call get_region some times, picking a higher target precision if the result is non-sensical.
    # By the end we have the range of energies to be used
    #####
    else:
        emin, emax = force_region
    eps = np.linspace(emin, emax, 5000)
    der  = d2g_DR(bR, bL, mR, mL, eps)
    filter = np.abs(der) > 10**(-11)
    eps  = eps[filter]
    der  = der[filter]
    prod = der[1:] * der[:-1]
    # We break the energy range into a grid with 5000 points (eps)
    # and store the (regularized) second derivative of g/Delta for these points (der)
    # We then filter out the points where der is essentially zero
    # The information of which intervals have sign changes is encoded in the sign of prod
    #####
    bracks = []
    for i in range(len(prod)):
        if prod[i] < 0:
            bracks.append((eps[i], eps[i+1]))
    # We store the brackets identified with prod in bracks
    #####
    d2_loc = lambda eps : d2g_DR(bR, bL, mR, mL, eps)   # localized version of d2g_DR to make it compatible with bracket
    d_loc  = lambda eps : dD_g(bR, bL, mR, mL, eps)     # this seems superfluous
    inflex = [bracket(d2_loc, b[0], b[1]) for b in bracks]
    return emin, emax, inflex
    # We run a "solve" in each of the brackets to get the inflection points, these are returned, together with the energy range

#checks if x is in ]a,b[. Note that if a > b this always fails (so we can represent empty intervals this way)
def is_in_interval(x, a, b):
    return x > a and x < b

# Auxiliary function that gives the derivatives of the contributions of a crossing
# between g/Delta and lam*x+eta happening at eps, with respect to the line parameters
def IFT_aux(bR, bL, mR, mL, eps, lam, eta):
    ER = np.exp(bR*(eps-mR))
    EL = np.exp(bL*(eps-mL))
    fR = 1 + ER
    fL = 1 + EL
    DE = EL - ER
    dD_D2 = (fR**2*EL*bL - fL**2*ER*bR)/DE**2
    dg_D2 = (bR*fL**3*ER*(1-ER) + bL*fR**3*EL*(1-EL))/(fR*fL*DE**2)
    _1_D  = fR*fL/DE
    return 1./(dg_D2 - lam*_1_D - dD_D2*(lam*eps + eta))

# Finds the root of a strictly monotonic function f. x0 is an intial guess and step is an initial step for the search
def expand_and_search(f, x0, step):
    a  = x0
    fa = f(a)
    fb = f(a + step)
    if fa*fb > 0 and np.abs(fb) > np.abs(fa):
        step *= -1
        fb = f(a + step)
    while fa*fb > 0:
        step *= 2
        fb = f(a + step)
    a, b = sorted((a, a + step))
    return bracket(f, a, b)

#######################
#       CLASSES       #
#######################


# Interval

# This class encapsulates the information necessary to obtain the crossings of a line with g/Delta
# in an interval where g/Delta has a well defined convexity and no singularities
# THIS SHOULD ONLY BE USED INTERNALLY BY THE Structure CLASS, TO ENSURE CONSISTENCE

# in the constructor the arguments are as follows:
#
# bR = Inverse Temperature in the Right Lead
# bL = Inverse Temperature in the Left Lead
# mR = Chemical Potential in the Right Lead
# mL = Chemical Potential in the Left Lead
# P1 = The point (eps1, gD1)
# P2 = The point (eps2, gD2)
#
# where:
# * eps1 and eps2 are the Energy values in the left and right endpoints of the interval respectively
# * gD1 and gD2 are the result of calling g_D(bR, bL, mR, mL, eps1) and g_D(bR, bL, mR, mL, eps1)
# respectively (which is done internally in the Structure class, in order to avoid repeating calculations)
class Interval:
    def __init__(self, P1, P2, bR, bL, mR, mL):
        #the curve interval from P1 to P2
        eps1, gD1 = P1
        eps2, gD2 = P2
        mid = 0.5*(eps1 + eps2)
        Del = Delta(bR, bL, mR, mL, mid)
        if Del > 0:
            self.type = "high"
        else:
            self.type = "low"
        #"high" means that
        # ax+b > g/Delta => is in boxcar
        #"low"  means that
        # ax+b < g/Delta => is in boxcar
        self.d1_loc = lambda eps : dg_D(bR, bL, mR, mL, eps)
        self.gD_loc = lambda eps : g_D(bR, bL, mR, mL, eps)
        d1_1 = dg_D(bR, bL, mR, mL, eps1)
        d1_2 = dg_D(bR, bL, mR, mL, eps2)
        d2   = d2g_DR(bR, bL, mR, mL, mid)
        if d1_1 * d1_2 > 0:
            d1 = dg_D(bR, bL, mR, mL, mid)
            if d1 * d2 > 0:
                self.extreme = gD1
            else:
                self.extreme = gD2
        #the extreme is in the border
        #find it using the monotonicity and the concavity in the interval
        else:
            ext = bracket(self.d1_loc, eps1, eps2)
            self.extreme = g_D(bR, bL, mR, mL, ext)
        #the extreme is in the interior (use the derivative to find it)
        self.a = (gD2 - gD1)/(eps2 - eps1)
        self.b = gD1 - self.a * eps1
        #the parameters of the line L1 that connects P1 and P2
        x = bracket(lambda eps : dg_D(bR, bL, mR, mL, eps) - self.a, eps1, eps2)
        self.c = g_D(bR, bL, mR, mL, x) - self.a * x
        #the tangent that is parallel to L1
        self.eps1 = eps1
        self.eps2 = eps2
        #the eps extrema of the interval
        if self.a == 0:
            self.ext1 = eps1
            self.ext2 = eps2
            self.tan1 = self.tan2 = eps1
            z1 = z2 = self.extreme
        else:
            cross = (self.extreme - self.c)/self.a
            if self.a * d2 > 0:
                self.ext1 = eps1
                self.tan1 = self.ext2 = cross
                self.tan2 = eps2
                z1 = self.extreme
                z2 = self.a*eps2 + self.c
            else:
                self.tan1 = eps1
                self.tan2 = self.ext1 = cross
                self.ext2 = eps2
                z1 = self.a*eps1 + self.c
                z2 = self.extreme
        I1 = sorted([z1, gD1])
        I2 = sorted([z2, gD2])
        self.Lvert1, self.Lvert2 = I1
        self.Rvert1, self.Rvert2 = I2
        D1 = sorted([d1_1, d1_2])
        self.d1_1, self.d1_2 = D1
        self.gD1 = gD1
        self.gD2 = gD2

    # return the crossings of LAM*eps + ETA with g/Delta in the interval
    def crossing(self, LAM, ETA):
        if LAM != self.a:
            if is_in_interval((self.b - ETA)/(LAM - self.a) , self.eps1, self.eps2):
                return [bracket(lambda eps: self.gD_loc(eps) - LAM*eps - ETA, self.eps1, self.eps2)]
        #does it cross the secant?
        if not (
            is_in_interval(LAM*self.eps1 + ETA, self.Lvert1, self.Lvert2) or
            is_in_interval(LAM*self.eps2 + ETA, self.Rvert1, self.Rvert2) or
            (LAM !=0 and is_in_interval((self.extreme - ETA)/LAM, self.ext1, self.ext2)) or
            (LAM != self.a and is_in_interval((self.b - ETA)/(LAM - self.a) , self.tan1, self.tan2))
            ):
            return []
        #does it miss the interval hitbox?
        if not is_in_interval(LAM, self.d1_1, self.d1_2):
            return []
        #if this is true any existing intersection should have been spotted
        x  = bracket(lambda eps: self.d1_loc(eps) - LAM, self.eps1, self.eps2)
        gD = self.gD_loc(x)
        if (gD - LAM * x - ETA)*(self.gD1 - LAM * self.eps1 - ETA) > 0:
            return []
        return [bracket(lambda eps: self.gD_loc(eps) - LAM * eps - ETA, self.eps1, x),
                bracket(lambda eps: self.gD_loc(eps) - LAM * eps - ETA, x, self.eps2)]

##################################################################################

# SingInterval

# This class has the same purpose as Interval, but for the case where one of the
# endpoints is a singularity
# THIS SHOULD ONLY BE USED INTERNALLY BY THE Structure CLASS, TO ENSURE CONSISTENCY

# in the constructor the arguments are as follows:
#
# bR = Inverse Temperature in the Right Lead
# bL = Inverse Temperature in the Left Lead
# mR = Chemical Potential in the Right Lead
# mL = Chemical Potential in the Left Lead
# P  = The point (epsP, gDP)
#
# where:
# * epsP is the Energy value in the endpoint of the interval that has no singularity
# * gDP is the result of calling g_D(bR, bL, mR, mL, epsP) (which is done internally in
# the Structure class, in order to avoid repeating calculations)
class SingInterval:
    def __init__(self, P, bR, bL, mR, mL):
        #the curve interval from P1 to P2
        epsP, gDP   = P
        eS          = (bR*mR - bL*mL)/(bR - bL)
        self.boundC = 1/(bL - bR)
        mid = 0.5*(epsP + eS)
        Del = Delta(bR, bL, mR, mL, mid)
        if Del > 0:
            self.type = "high"
        else:
            self.type = "low"
        #"high" means that
        # ax+b > g/Delta => is in boxcar
        #"low"  means that
        # ax+b < g/Delta => is in boxcar
        self.d1_loc = lambda eps : dg_D(bR, bL, mR, mL, eps)
        self.gD_loc = lambda eps : g_D(bR, bL, mR, mL, eps)
        if eS > epsP:
            self.sing = "right"
            eps1 = epsP
            eps2 = eS
            gD1  = gDP
            gD2  = INF * Del
            d1_1 = dg_D(bR, bL, mR, mL, eps1)
            d1_2 = INF * Del
        else:
            self.sing = "left"
            eps1 = eS
            eps2 = epsP
            gD1  = INF * Del
            gD2  = gDP
            d1_1 = -INF * Del
            d1_2 = dg_D(bR, bL, mR, mL, eps2)
        #on which side of the interval is the singularity?
        d2 = d2g_DR(bR, bL, mR, mL, mid)
        if d1_1 * d1_2 > 0:
            if self.sing == "right":
                self.extreme = gD1
            else:
                self.extreme = gD2
        #the extreme is in the border
        #find it using the monotonicity and the concavity in the interval
        else:
            cross = eS + self.boundC/gDP
            if self.sing == "left":
                ext = bracket(self.d1_loc, cross, eps2)
            else:
                ext = bracket(self.d1_loc, eps1, cross)
            self.extreme = g_D(bR, bL, mR, mL, ext)
        #the extreme is in the interior (use the derivative to find it and the bound to avoid the singularity)
        cross = eS + self.boundC/self.extreme
        if self.sing == "left":
            self.ext1 = cross
            self.ext2 = eps2
        else:
            self.ext1 = eps1
            self.ext2 = cross
        self.epsP = epsP
        self.gDP  = gDP
        self.gDmin , self.gDmax = sorted([gD1, gD2])
        self.eps1 , self.eps2 = sorted([epsP, eS])
        D1 = sorted([d1_1, d1_2])
        self.d1_1, self.d1_2 = D1
        self.eS = eS
        self.gD1 = gD1
        self.gD2 = gD2

    # return the crossings of LAM*eps + ETA with g/Delta in the interval
    def crossing(self, LAM, ETA):
        if is_in_interval(LAM*self.epsP + ETA , self.gDmin, self.gDmax):
            bcross = self.__bound_crossing(LAM, ETA)[0]
            if self.sing == "left":
                return [bracket(lambda eps: self.gD_loc(eps) - LAM*eps - ETA, bcross, self.eps2)]
            return [bracket(lambda eps: self.gD_loc(eps) - LAM*eps - ETA, self.eps1, bcross)]
        #does it cross the secant?
        if not is_in_interval(LAM, self.d1_1, self.d1_2):
            return []
        if not (LAM !=0 and is_in_interval((self.extreme - ETA)/LAM, self.eps1, self.eps2)):
            g1, g2 = sorted([self.gDP, self.extreme])
            if not is_in_interval(LAM*self.epsP + ETA , g1, g2):
                return []
        bcross = self.__bound_crossing(LAM, ETA)
        if len(bcross) == 0:
            return []
        #does it miss the interval hitbox?
        if len(bcross) == 2:
            b1, b2 = bcross
        else:
            if self.sing == "left":
                b1 = bcross[0]
                b2 = self.eps2
            else:
                b1 = self.eps1
                b2 = bcross[0]
        #if there are any crossings then they are all in the [b1, b2] interval
        try:
            x  = bracket(lambda eps: self.d1_loc(eps) - LAM, b1, b2)
        except ValueError:
            return []
        #this catches the situation where the point at which the tangent touches the curve
        #is outside of the bracket
        gD = self.gD_loc(x)
        if (gD - LAM * x - ETA)*(self.gDP - LAM * self.epsP - ETA) > 0:
            return []
        return [bracket(lambda eps: self.gD_loc(eps) - LAM * eps - ETA, b1, x),
                bracket(lambda eps: self.gD_loc(eps) - LAM * eps - ETA, x, b2)]

    # Auxiliary function used internally by crossing. It checks the crossings between
    # LAM*eps + ETA and the bound (1/db(eS-eps)) for the function around a singularity
    # (in order to avoid having to directly deal with the singularity)
    def __bound_crossing(self, LAM, ETA):
        a = LAM
        b = ETA - LAM*self.eS
        c = -self.boundC - ETA*self.eS
        if a == 0:
            return (-c/b,)
        d = b**2 - 4*a*c
        if d <= 0:
            return ()
        #double roots here never yield true crossings, so we can just ignore them
        D = np.sqrt(d)/(2*a)
        M = -b/(2*a)
        sols = sorted((M+D, M-D))
        return [s for s in sols if is_in_interval(s, self.eps1, self.eps2)]

##################################################################################

# Structure

# This abstracts the curve g/Delta and exposes an interface for obtaining I, J and Var(I)
# for the boxcar corresponding to a given line

#in the constructor the arguments are as follows:
#
# bR   = Inverse Temperature in the Right Lead
# bL   = Inverse Temperature in the Left Lead
# mR   = Chemical Potential in the Right Lead
# mL   = Chemical Potential in the Left Lead
# prec = Precision (Absolute Error Tolerance)
class Structure:
    def __init__(self, bR, bL, mR, mL, prec, force_region = None):
        emin, emax, inflex = get_structure(bR, bL, mR, mL, prec, force_region)
        self.region = (emin, emax)
        inflex += [emin, emax]
        db  = bR - bL
        dbm = bR*mR - bL*mL
        if bR == bL:
            self.extreme = self.__extreme_symmetric
        else:
            eS = dbm/db
            self.eS = eS
            if db > 0:
                self.extreme = self.__extreme_positive
            else:
                self.extreme = self.__extreme_negative
        if (dbm - emin*db)*(dbm - emax*db) < 0:
            singular = True
            inflex.append(eS)
            if is_in_interval(eS - 4./db, emin, emax):
                inflex.append(eS - 4./db)
            if is_in_interval(eS + 4./db, emin, emax):
                inflex.append(eS + 4./db)
        else:
            singular = False
        inflex.sort()
        intervals = [(inflex[i], inflex[i+1]) for i in range(len(inflex) - 1)]
        P = lambda eps: (eps, g_D(bR, bL, mR, mL, eps))
        self.intervals = []
        if singular:
            for inter in intervals:
                if eS not in inter:
                    self.intervals.append(Interval(P(inter[0]), P(inter[1]), bR, bL, mR, mL))
                elif inter[0] == eS:
                    self.intervals.append(SingInterval(P(inter[1]), bR, bL, mR, mL))
                else:
                    self.intervals.append(SingInterval(P(inter[0]), bR, bL, mR, mL))
        else:
            self.intervals = [Interval(P(inter[0]), P(inter[1]), bR, bL, mR, mL) for inter in intervals]
        self.bR, self.bL, self.mR, self.mL = bR, bL, mR, mL
        self.TR, self.TL = 1./self.bR , 1./self.bL
        self.IJVfull  = self.__IJV(None, None)
        self.singular = singular
        self.dbm, self.db = dbm, db
        self.mb = (self.mR + self.mL)/2
        self.gDemin = self.intervals[0].gD1
        self.gDemax = self.intervals[-1].gD2

    # returns the array (I, J, Var(I)) corresponding to the boxcar [a, b]
    # this is only used internally (it's where the integrals are actually calculated)
    #
    # None can be passed a value for a and b, meaning the corresponding endpoint of the energy range
    def __IJV(self, a, b):
        #Note that spence(1-z) = Li_2(z) (i.e. a polylog)
        if a is not None:
            ERa = np.exp(self.bR*(a - self.mR))
            ELa = np.exp(self.bL*(a - self.mL))
            Li2Ra = dilog(1 + 1./ERa)
            Li2La = dilog(1 + 1./ELa)
            lfRa = - np.log(1 + ERa)
            lfLa = - np.log(1 + ELa)
            Ia = self.TR*lfRa - self.TL*lfLa
            Ja = (a*(a-self.mR) + self.TR*(a*lfRa + self.TR*Li2Ra)) - (a*(a-self.mL) + self.TL*(a*lfLa + self.TL*Li2La))
            Va = - self.TR/(1 + ERa) - self.TL/(1 + ELa)
        else:
            Ia = 0
            Ja = pi2_6*(self.TL**2-self.TR**2) + (self.mL**2 - self.mR**2)/2
            Va = -(self.TL + self.TR)
        if b is not None:
            ERb = np.exp(self.bR*(b - self.mR))
            ELb = np.exp(self.bL*(b - self.mL))
            Li2Rb = dilog(1 + 1./ERb)
            Li2Lb = dilog(1 + 1./ELb)
            lfRb = - np.log(1 + ERb)
            lfLb = - np.log(1 + ELb)
            Ib = self.TR*lfRb - self.TL*lfLb
            Jb = (b*(b-self.mR) + self.TR*(b*lfRb + self.TR*Li2Rb)) - (b*(b-self.mL) + self.TL*(b*lfLb + self.TL*Li2Lb))
            Vb = - self.TR/(1 + ERb) - self.TL/(1 + ELb)
        else:
            Ib = self.mR - self.mL
            Jb = Vb = 0
        return np.array([Ib - Ia, Jb - Ja, Vb - Va])

    # returns an array [(a_1, b_1), (a_2, b_2), ..., (a_N, b_N)] representing the boxcar collection
    # [a_1, b_1], [a_2, b_2], ... [a_N, b_N] defined by the line A*eps + B.
    #
    # a value None indicates +/- inf depending on whether it occurs at a_1 or b_N
    def boxcars(self, A, B):
        crossings = []
        emin, emax = self.region
        for inter in self.intervals:
            crossings += inter.crossing(A, B)
        crossings = [emin] + sorted(crossings) + [emax]
        #sort and add the "extremes"
        I0 = self.intervals[0]
        if (I0.type == "high" and A*I0.eps1 + B > I0.gD1) or (I0.type == "low" and A*I0.eps1 + B < I0.gD1):
            leftin = True
        else:
            leftin = False
        #"high" means that
        # ax+b > g/Delta => is in boxcar
        #"low"  means that
        # ax+b < g/Delta => is in boxcar
        if leftin:
            boxes = [(crossings[2*n], crossings[2*n + 1]) for n in range(len(crossings)//2)]
        else:
            boxes = [(crossings[2*n + 1], crossings[2*n + 2]) for n in range((len(crossings) - 1)//2)]
        return boxes

    # returns the array (I, J, Var(I)) corresponding to the boxcar collection defined by the line A*eps + B
    # This method is the main interface of the Structure object with outside code
    def IJV(self, A, B):
        boxes = self.boxcars(A, B)
        if len(boxes) != 0:
            return sum([self.__IJV(a,b) for a,b in boxes])
        return np.array([0,0,0])

    # These are internal methods. One of these is picked as self.extreme during the object creation
    #
    # All the functions return an array (I1, J1, V1, I2, J2, V2), where J1 is the smallest value for J
    # possible, given that I = I1 and J2 is the largest value for J possible, given that I = I2
    # V1 and V2 are the minimal variances for (I1, J1) and (I2, J2). This is useful to graph all the
    # possible values of (I, J) (by graphing the boundary of the convex region they define)
    #
    # The argument a is such that if we sweep through all the values in self.region we obtain
    # all the values with the precision provided during the object creation
    def __extreme_negative(self, a):      # this assumes delta(beta) < 0
        if a > self.eS:
            ijv = self.__IJV(self.eS, a)
        else:
            ijv = self.__IJV(a, self.eS)
        return np.concatenate((ijv, self.IJVfull - ijv)) # the calculated boxcar gives the minimum

    def __extreme_symmetric(self, a):     # this assumes delta(beta) = 0
        if self.dbm > 0:
            ijv = self.__IJV(a, None)
        else:
            ijv = self.__IJV(None, a)
        return np.concatenate((self.IJVfull - ijv, ijv)) # the calculated boxcar gives the maximum

    def __extreme_positive(self, a):      # this assumes delta(beta) > 0
        if a > self.eS:
            ijv = self.__IJV(self.eS, a)
        else:
            ijv = self.__IJV(a, self.eS)
        return np.concatenate((self.IJVfull - ijv, ijv)) # the calculated boxcar gives the maximum


    # The following are convenience methods that automate certain operations:

    # Returns a tuple with the range of values I can take
    def range_I(self):
        emin, emax = self.region
        ext_point  = self.extreme(emin)
        return (ext_point[0], ext_point[3])

    # Returns a tuple with the range of values J can take
    def range_J(self):
        ext_point  = self.extreme(0)
        return (ext_point[1], ext_point[4])

    # Graphs the contour of the (I,J) region and returns the raw data
    #
    # TOL gives a tolerance for the extreme corners of the graph to be clipped (for performance and smoothness)
    def get_IJ_extreme(self, TOL = 0.001, points = 200):
        tol = np.log(TOL)
        emin, emax = self.region
        emin = max(emin, min(self.mR + tol/self.bR, self.mL + tol/self.bL))
        emax = min(emax, max(self.mR - tol/self.bR, self.mL - tol/self.bL))
        erange = np.linspace(emin, emax, points)
        data = np.array([self.extreme(eps) for eps in erange]).T
        return data

    def get_VIJ_extreme(self, TOL = 0.001, points = 200):
        tol = np.log(TOL)
        emin, emax = self.region
        emin = max(emin, min(self.mR + tol/self.bR, self.mL + tol/self.bL))
        emax = min(emax, max(self.mR - tol/self.bR, self.mL - tol/self.bL))
        erange = np.linspace(emin, emax, points)
        data = np.array([self.extreme(eps) for eps in erange]).T
        return data

    def IJ_derivatives(self, LAM, ETA):
        bR, bL, mR, mL = self.bR, self.bL, self.mR, self.mL
        boxes = self.boxcars(LAM, ETA)
        eps_plus  = [b[1] for b in boxes if b[1] is not None]
        eps_minus = [b[0] for b in boxes if b[0] is not None]
        IFT_plus  = [(eps, IFT_aux(bR, bL, mR, mL, eps, LAM, ETA)) for eps in eps_plus]
        IFT_minus = [(eps, IFT_aux(bR, bL, mR, mL, eps, LAM, ETA)) for eps in eps_minus]
        dI = sum([I[1] for I in IFT_plus]) - sum([I[1] for I in IFT_minus])
        dJ = sum([I[0]**2*I[1] for I in IFT_plus]) - sum([I[0]**2*I[1] for I in IFT_minus])
        return dI, dJ

    # REFACTOR THE CODE THAT CALCULATES I, J and V TO MAKE THIS LESS INEFFICIENT
    def boxcar_solve(self, I_target, J_target, tol):
        lam, eta = -self.db, 0 #initial conditions that never give trivial boxcars (MELHORAR ISSO)
        I, J, V = self.IJV(lam, eta)
        flag    = 0 # 0 for I, 1 for J
        while np.abs(I - I_target) > tol or np.abs(J - J_target) > tol:
            if flag == 0:
                I_loc = lambda x : self.IJV(lam, x)[0] - I_target
                dI, dJ = self.IJ_derivatives(lam, eta)
                if np.abs(dI) > 10**-10:
                    step = (I_target - I)/dI
                else:
                    if I_target > I:
                        step = tol
                    else:
                        step = -tol
                eta  = expand_and_search(I_loc, eta, step)
                flag = 1
            else:
                J_loc = lambda x : self.IJV(x, eta)[1] - J_target
                dI, dJ = self.IJ_derivatives(lam, eta)
                if np.abs(dJ) > 10**-10:
                    step = (J_target - J)/dJ
                else:
                    if J_target > J:
                        step = tol
                    else:
                        step = -tol
                lam  = expand_and_search(J_loc, lam, step)
                flag = 0
            I, J, V = self.IJV(lam, eta)
        return lam, eta, self.boxcars(lam, eta)

##################################################################################

def bifurcations(br, bl, mr, ml, a, b):
    if br != bl:
        eS = (br*mr - bl*ml)/(br - bl)
        if a > eS or b < eS:
            z = [np.linspace(a+.001, b-.001, 500)]
        else:
            z = [np.linspace(a+.001, eS-.001, 500), np.linspace(eS+.001, b-.001, 500)]
    else:
        z = [np.linspace(a, b, 500)]
    z   = [zz[np.newaxis] for zz in z]
    eta = np.linspace(-5, 5, 1000)[np.newaxis]
    return ([np.concatenate((dg_D(br, bl, mr, ml, zz), g_D(br, bl, mr, ml, zz) - zz*dg_D(br, bl, mr, ml, zz))) for zz in z] +
    [np.concatenate((np.zeros(eta.shape), eta))])

def bifurcations_IJ(struct):
    a, b = struct.region
    br, bl, mr, ml = struct.bR, struct.bL, struct.mR, struct.mL
    data   = bifurcations(br, bl, mr, ml, a, b)
    retval = []
    for curve in data:
        params = curve.transpose()
        IJ     = []
        for lam, eta in params:
            I, J, V = struct.IJV(lam, eta)
            IJ.append([I, J])
        IJ = np.array(IJ).transpose()
        retval.append(IJ)
    #extremes = struct.get_IJ_extreme()
    #return retval + [np.concatenate((extremes[0][np.newaxis], extremes[1][np.newaxis])), np.concatenate((extremes[3][np.newaxis], extremes[4][np.newaxis]))]
    return retval

#######################
#       The GUI       #
#######################

br, bl, mr, ml = .5, 1., 0., 1.
S = None
fig, (ax_ij, ax_box) = plt.subplots(1,2)
plt.ion()
plt.show()

def submit1(data):
    global br
    br = 1./float(data)

def submit2(data):
    global bl
    bl = 1./float(data)

def submit3(data):
    global mr
    mr = float(data)

def submit4(data):
    global ml
    ml = float(data)

def update(event):
    global S
    ax_ij.clear()
    ax_box.clear()
    plt.title("Processing ...")
    fig.canvas.draw()
    fig.canvas.flush_events()
    S = Structure(br, bl, mr, ml, 10**(-6))
    boundary = S.get_IJ_extreme(10**-3)
    ax_ij.plot(boundary[0], boundary[1])
    ax_ij.plot(boundary[3], boundary[4])
    curves   = bifurcations_IJ(S)
    for curve in curves:
        ax_ij.plot(curve[0], curve[1])
    plt.title("Done!")
    fig.canvas.draw()
    fig.canvas.flush_events()

#
# THE BUTTONS MIGHT APPEAR TO BE FLIPPED, BUT THIS IS DONE BECAUSE THE CODE WAS WRITTEN USING THE CONVENTIONS
# Delta f = f_R - f_L, delta_beta = beta_R - beta_L, delta_mu = mu_R - mu_L, etc
# FLIPPING THE INPUTS MAKES THE END RESULTS FOLLOW THE CONVENTION USED IN THE PAPER (THAT IS ALSO THE MORE COMMONPLACE ONE)
#

fig.subplots_adjust(bottom=0.2)
axbox1 = fig.add_axes([0.1, 0.05, 0.1, 0.075])
TLbox  = TextBox(axbox1, "$T_L$")
TLbox.on_submit(submit1)
TLbox.set_val("2.0")
axbox2 = fig.add_axes([0.25, 0.05, 0.1, 0.075])
TRbox  = TextBox(axbox2, "$T_R$")
TRbox.on_submit(submit2)
TRbox.set_val("1.0")
axbox3 = fig.add_axes([0.4, 0.05, 0.1, 0.075])
MLbox  = TextBox(axbox3, r"$\mu_L$")
MLbox.on_submit(submit3)
MLbox.set_val("0.0")
axbox4 = fig.add_axes([0.55, 0.05, 0.1, 0.075])
MRbox  = TextBox(axbox4, r"$\mu_R$")
MRbox.on_submit(submit4)
MRbox.set_val("1.0")
axbox5 = fig.add_axes([0.7, 0.05, 0.2, 0.075])
Ubutton = Button(axbox5, "Update")
Ubutton.on_clicked(update)

def onclick_graph(event):
    I, J = event.xdata, event.ydata
    if event.inaxes != ax_ij:
        return
    plt.title("Processing ...")
    fig.canvas.draw()
    fig.canvas.flush_events()
    lam, eta, box = S.boxcar_solve(I, J, 10**-3)
    xbox = np.array([S.region[0]] + [box[i//4][(i%4)//2] for i in range(4*len(box))] + [S.region[1]])
    ybox = np.array([0]+([0., 1., 1., 0.]*len(box)) + [0])
    ax_box.clear()
    ax_box.set_xlim(S.region[0], S.region[1])
    ax_box.set_ylim(-0.1, 1.1)
    ax_box.plot(xbox, ybox)
    plt.title("Done!")
    fig.canvas.draw()
    fig.canvas.flush_events()

cid = fig.canvas.mpl_connect('button_press_event', onclick_graph)
