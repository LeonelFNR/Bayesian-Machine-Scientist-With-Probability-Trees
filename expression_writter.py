from sympy import *

def round_expr(expr, num_digits):
    return expr.xreplace({n : round(n, num_digits) for n in expr.atoms(Number)})
    
    
def latex_val(t):
    cansp = sympify(str(t).replace('_', ''))
    cansp = cansp.subs(dict([(str(p).replace('_', ''), v) for p, v in t.par_values['d0'].items()]))
    #cansp = sympy.simplify(cansp)
    return latex(round_expr(cansp, 2))

def simplificador_prova(t):
    try:
        # Simplificació algebraica, passem primer a string
        expr = sympify(str(t))
        # Substitució constants, creació diccionari entre constants i valors respectius
        expr_mod = expr.subs(dict([(p, v) for p, v in t.par_values['d0'].items()]))
        #expr_mod = simplify(expr_mod)
        return latex(round_expr(expr_mod, 2))
    except SympifyError as e:
        print(f"SympifyError: {e}")
        return latex(round_expr(expr_mod, 2))
    except Exception as e:
        print(f"Unexpected error: {e}")
        return str(expr)
    