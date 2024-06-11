import math

def unit_au2ang():
    return 0.529177249

def unit_ang2au():
    return 1.0/unit_au2ang()

def unit_ang2meter():
    return 1.0*10**-10
def unit_meter2ang():
    return 1.0/unit_ang2meter()

def unit_hartree2kcal():
    return 627.51
def unit_kcal2hartree():
    return 1.0/unit_hartree2kcal()

def unit_hartree2kJ():
    return 2625.5

def unit_kJ2hartree():
    return 1.0/unit_hartree2kJ()

def unit_deg2rad():
    return 2 * math.pi / 360

def unit_rad2deg():
    return 1.0 / unit_deg2rad()