# parse_example.py
"""
Improved parsing pipeline with required "must-do" changes:
 - Adds JSON-serializable outputs (sympy_str, sympy_latex, srepr, ast, free_symbols, canonical_str)
 - Adds `backend` (which parser succeeded) and `success` fields
 - Adds a conservative implicit-multiplication inserter to avoid token concatenation bugs
 - Keeps original fallbacks (parse_latex, latex2sympy2, ascii sympify)
 - Returns structured `notes` for traceability
"""
import re
import sympy as sp
from sympy import sympify
# parse_latex may raise if antlr isn't installed; that's handled in try/except at call sites
try:
    from sympy.parsing.latex import parse_latex
except Exception:
    parse_latex = None

# Try to import latex2sympy2 (if installed) and detect usable functions
latex2sympy2_module = None
latex2sympy2_func = None
try:
    import latex2sympy2 as l2s2
    latex2sympy2_module = l2s2
    # Try common function names used historically
    for fname in ("latex2sympy", "latex2sympy2", "process_sympy", "latex2sympy2sympy"):
        if hasattr(l2s2, fname):
            latex2sympy2_func = getattr(l2s2, fname)
            break
    # some versions expose function at top-level as callable
    if latex2sympy2_func is None and callable(l2s2):
        latex2sympy2_func = l2s2
except Exception:
    latex2sympy2_module = None
    latex2sympy2_func = None


def recursive_frac_replace(s: str) -> str:
    """
    Replace nested \frac{a}{b} with ((a)/(b)) recursively until none remain.
    Handles one-level braces but recursively reduces nested fractions. Conservative.
    """
    pattern = re.compile(r'\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}')
    prev = None
    cur = s
    max_iters = 100
    it = 0
    while prev != cur and it < max_iters:
        prev = cur
        cur = pattern.sub(lambda m: f"(({m.group(1)})/({m.group(2)}))", cur)
        it += 1
    return cur


def insert_implicit_multiplication(s: str) -> str:
    
    
    # Conservative insertion of '*' for implicit multiplication to avoid token merging problems
    # (e.g., turn '2x' into '2*x', '2(' -> '2*(', ')2' -> ')*2', ')\x' -> ')*\x' etc).
    # Keep function names like sin(x) intact (do not insert between letters and '(').
    # This is deliberately conservative and performed after the main LaTeX->ASCII substitutions.
    # Note: do NOT collapse '**' sequences (exponent operator) — keep them intact.
    
    
    t = s

    # 1) Digit followed by backslash (e.g., "2\frac" -> "2*\frac")
    t = re.sub(r'(?<=\d)(?=\\)', '*', t)

    # 2) Digit followed by letter (2x -> 2*x)
    t = re.sub(r'(?<=\d)(?=[A-Za-z])', '*', t)

    # 3) Digit followed by '('  (2( -> 2*()
    t = re.sub(r'(?<=\d)(?=\()', '*', t)

    # 4) ')' followed by digit or backslash or letter -> )*x, )*2, )*\frac
    t = re.sub(r'(?<=\))(?=(\\|\d|[A-Za-z]))', '*', t)

    # 5) ']' or '}' followed by digit/letter/backslash -> add '*'
    t = re.sub(r'(?<=[\]\}])(?=(\\|\d|[A-Za-z]))', '*', t)

    # 6) letter or digit followed by '[' or '{' -> add *  (x[ -> x*[ ; 2{ -> 2*{)
    t = re.sub(r'(?<=[A-Za-z0-9\)])(?=(\{|\[))', '*', t)

    # 7) handle adjacent closing and opening parentheses: ")(" -> ")*("
    t = t.replace(')(', ')*(')

    # 8) tidy spaces (do not modify * sequences)
    t = re.sub(r'\s+', ' ', t).strip()

    return t


def latex_to_ascii_robust(latex: str) -> str:
    """
    More robust LaTeX -> ASCII conversion:
     - recursively convert \frac{...}{...}
     - convert ^{...} -> **(...)
     - convert \cdot, \times to *
     - convert trig commands \sin -> sin
     - remove $ and small spacing macros
     - insert implicit multiplication conservatively
    Note: conservative — not a full LaTeX parser.
    """
    s = latex.strip()
    # remove $ delimiters
    s = s.replace('$', '')
    # remove small spacing macros (conservative)
    s = re.sub(r'\\[,~\s]+', '', s)

    # handle \left( \right) -> ( )
    s = s.replace(r'\left(', '(').replace(r'\right)', ')')
    s = s.replace(r'\left[', '[').replace(r'\right]', ']')
    s = s.replace(r'\left\{', '{').replace(r'\right\}', '}')

    # recursively replace \frac
    s = recursive_frac_replace(s)

    # replace common commands
    s = s.replace(r'\cdot', '*').replace(r'\times', '*')
    s = s.replace(r'\pi', 'pi')

    # convert trig latex names \sin -> sin (sympy recognizes sin)
    s = re.sub(r'\\(sin|cos|tan|sec|csc|cot|exp|log|ln|sqrt)', r'\1', s)

    # convert superscript with braces: ^{...} -> **(...)
    s = re.sub(r'\^\{([^{}]+)\}', r'**(\1)', s)
    # convert superscript without braces a^b -> a**b (cautious)
    s = re.sub(r'([0-9a-zA-Z\)\]])\^([0-9a-zA-Z\(])', r'\1**\2', s)

    # replace remaining ^ with **
    s = s.replace('^', '**')

    # ensure explicit multiplication in common implicit cases (e.g., 2x, 2(x), )x, )2)
    s = insert_implicit_multiplication(s)

    # final tidy
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def detect_derivative_pattern(latex: str):
    """
    Detect variants of \frac{d}{dx} f(x) or d/dx f(x) and return (match_kind, func_str, var_str, order, deriv_type)
    Returns None if not derivative-like.

    Currently supports:
     - \frac{d}{dx}   -> order 1 total derivative
     - \frac{d^n}{dx^n} -> order n total derivative
     - \frac{\partial}{\partial x} -> partial derivative
     - d/dx f(x)
    """
    s = latex.strip()

    # \frac{d^n}{dx^n} pattern (total derivative, n optional)
    m = re.match(r'\\frac\s*\{\s*d\^?(\d*)\s*\}\s*\{\s*d([a-zA-Z0-9_]+)\^?(\d*)\s*\}\s*(.+)', s)
    if m:
        # orders may be present in the numerator or denominator
        num_order = m.group(1) or ''
        den_var = m.group(2)
        den_order = m.group(3) or ''
        order = 1
        if num_order:
            order = int(num_order)
        elif den_order:
            order = int(den_order)
        func = m.group(4).strip()
        return ("latex_frac_order", func, den_var, order, "total")

    # \frac{d}{dx} f(x) simpler (explicit dx)
    m2 = re.match(r'\\frac\s*\{\s*d\s*\}\s*\{\s*d([a-zA-Z0-9_]+)\s*\}\s*(.+)', s)
    if m2:
        var = m2.group(1)
        func = m2.group(2).strip()
        return ("latex_frac", func, var, 1, "total")

    # partial derivative: \frac{\partial^k}{\partial x^k} f
    m3 = re.match(r'\\frac\s*\{\s*\\partial\^?(\d*)\s*\}\s*\{\s*\\partial([a-zA-Z0-9_]+)\^?(\d*)\s*\}\s*(.+)', s)
    if m3:
        num_order = m3.group(1) or ''
        den_var = m3.group(2)
        den_order = m3.group(3) or ''
        order = 1
        if num_order:
            order = int(num_order)
        elif den_order:
            order = int(den_order)
        func = m3.group(4).strip()
        return ("latex_partial_frac", func, den_var, order, "partial")

    # non-latex pattern: d/dx f(x)
    m4 = re.match(r'd\s*/\s*d([a-zA-Z0-9_]+)\s*(.+)', s)
    if m4:
        var = m4.group(1)
        func = m4.group(2).strip()
        return ("d_slash", func, var, 1, "total")

    return None


def try_latex2sympy2(expr: str):
    """Attempt to convert using latex2sympy2 if available; return (sympy_expr or None, notes_list)"""
    notes = []
    if latex2sympy2_func is None:
        notes.append({"level": "info", "msg": "latex2sympy2 not available"})
        return None, notes
    try:
        out = latex2sympy2_func(expr)
        # The external function may return a SymPy object or a string of SymPy code
        if isinstance(out, str):
            try:
                return sympify(out), notes + [{"level": "info", "msg": "latex2sympy2 returned string and sympify succeeded"}]
            except Exception:
                try:
                    return sp.parse_expr(out), notes + [{"level": "info", "msg": "latex2sympy2 returned string and parse_expr succeeded"}]
                except Exception:
                    notes.append({"level": "warn", "msg": "latex2sympy2 returned string but could not be sympified/parsed"})
                    return None, notes
        # if it already is a sympy object
        if hasattr(out, "free_symbols"):
            return out, notes + [{"level": "info", "msg": "latex2sympy2 returned sympy object"}]
        # unknown return type
        notes.append({"level": "warn", "msg": "latex2sympy2 returned unknown type"})
        return None, notes
    except Exception as e:
        notes.append({"level": "warn", "msg": f"latex2sympy2 attempt failed: {e}"})
        return None, notes


def expr_to_simple_ast(e):
    """
    Convert a sympy expression into a small JSON-friendly AST.
    Covers common node types: Number, Symbol, Add, Mul, Pow, Function, Div (represented as Mul with Rational),
    and fallback for unknown node types.
    """
    # Atoms
    if e is None:
        return None
    if getattr(e, "is_Number", False):
        # Use string form for numbers to preserve e.g. Rational('1/2')
        return {"node": "Number", "value": str(e)}
    if getattr(e, "is_Symbol", False):
        return {"node": "Symbol", "name": str(e)}
    # Composite nodes
    if getattr(e, "is_Add", False):
        return {"node": "Add", "args": [expr_to_simple_ast(a) for a in e.args]}
    if getattr(e, "is_Mul", False):
        return {"node": "Mul", "args": [expr_to_simple_ast(a) for a in e.args]}
    if getattr(e, "is_Pow", False):
        return {"node": "Pow", "base": expr_to_simple_ast(e.base), "exp": expr_to_simple_ast(e.exp)}
    if getattr(e, "is_Function", False) or hasattr(e, "func"):
        # Function or other callable-like objects
        func_name = getattr(e.func, "__name__", str(e.func))
        return {"node": "Function", "name": func_name, "args": [expr_to_simple_ast(a) for a in e.args]}
    # fallback: return string
    return {"node": "Expr", "str": str(e)}


def serialize_result(result: dict) -> dict:
    """
    Take the intermediate result (with sympy objects possibly present) and add JSON-serializable fields:
      - sympy_str/sympy_latex/sympy_srepr/canonical_str/free_symbols/ast
      - derived_str/derived_latex/derived_srepr
      - backend, success
    Also ensures notes exist as a list.
    """
    result = dict(result)  # shallow copy to avoid side-effects
    notes = result.get("notes", [])
    if notes is None:
        notes = []
    # ensure notes is a list of dicts
    normalized_notes = []
    for n in notes:
        if isinstance(n, dict):
            normalized_notes.append(n)
        else:
            normalized_notes.append({"level": "info", "msg": str(n)})
    result["notes"] = normalized_notes

    sym = result.get("sympy", None)
    backend = result.get("backend", None)
    if sym is not None:
        try:
            result["sympy_str"] = str(sym)
            try:
                result["sympy_latex"] = sp.latex(sym)
            except Exception:
                result["sympy_latex"] = None
            try:
                result["sympy_srepr"] = sp.srepr(sym)
            except Exception:
                result["sympy_srepr"] = None
            try:
                result["free_symbols"] = sorted([str(s) for s in sym.free_symbols])
            except Exception:
                result["free_symbols"] = []
            try:
                result["ast"] = expr_to_simple_ast(sym)
            except Exception:
                result["ast"] = None
            try:
                result["canonical_str"] = str(sp.simplify(sym))
            except Exception:
                result["canonical_str"] = str(sym)
            result["backend"] = backend or "unknown"
            result["success"] = True
        except Exception as e:
            result["success"] = False
            result.setdefault("notes", []).append({"level": "error", "msg": f"serialization_failed: {e}"})
    else:
        result["sympy_str"] = None
        result["sympy_latex"] = None
        result["sympy_srepr"] = None
        result["free_symbols"] = []
        result["ast"] = None
        result["canonical_str"] = None
        result["backend"] = backend or "failed"
        result["success"] = False

    # derived
    d = result.get("derived", None)
    if d is not None:
        try:
            result["derived_str"] = str(d)
            try:
                result["derived_latex"] = sp.latex(d)
            except Exception:
                result["derived_latex"] = None
            try:
                result["derived_srepr"] = sp.srepr(d)
            except Exception:
                result["derived_srepr"] = None
        except Exception:
            result["derived_str"] = None
            result["derived_latex"] = None
            result["derived_srepr"] = None
    else:
        result["derived_str"] = None
        result["derived_latex"] = None
        result["derived_srepr"] = None

    # keep original but ensure it's a string
    result["original"] = str(result.get("original", ""))

    return result


def parse_expression(expr: str, compute_derivative: bool = True, mode_hint: str = None) -> dict:
    """
    Main entrypoint.
    - If derivative pattern detected, parse inner function and compute derivative (if compute_derivative=True)
    - Try parse_latex (if available)
    - Try latex2sympy2 (if installed)
    - Fallback to robust ascii conversion + sympify
    Returns a dict with serializable fields (see serialize_result).
    """
    result = {"original": expr, "kind": None, "sympy": None, "derived": None, "ascii_fallback": None, "notes": [], "backend": None}

    # detect derivative-like input
    deriv = detect_derivative_pattern(expr)
    if deriv:
        kind, func_str, var, order, deriv_type = deriv
        result["kind"] = "derivative"
        result["notes"].append({"level": "info", "msg": f"Detected derivative pattern variant={kind}, var={var}, order={order}, type={deriv_type}"})
        # parse inner function robustly
        inner = None
        backend_used = None

        # 1) try parse_latex (if available and looks like latex)
        if parse_latex is not None:
            try:
                inner = parse_latex(func_str)
                backend_used = "parse_latex"
                result["notes"].append({"level": "info", "msg": "parse_latex succeeded on inner function"})
            except Exception as e:
                result["notes"].append({"level": "info", "msg": f"parse_latex failed on inner function: {e}"})

        # 2) try latex2sympy2
        if inner is None and latex2sympy2_func is not None:
            parsed, notes = try_latex2sympy2(func_str)
            result["notes"].extend(notes)
            if parsed is not None:
                inner = parsed
                backend_used = "latex2sympy2"

        # 3) fallback ascii conversion
        if inner is None:
            try:
                ascii_conv = latex_to_ascii_robust(func_str)
                ascii_conv = ascii_conv.strip()
                result["ascii_fallback"] = ascii_conv
                inner = sympify(ascii_conv)
                backend_used = "sympify_ascii"
                result["notes"].append({"level": "info", "msg": "sympify succeeded on ascii-fallback of inner function"})
            except Exception as e:
                result["notes"].append({"level": "warn", "msg": f"sympify on ascii-fallback failed: {e}"})
                # last attempt: sympify original raw string
                try:
                    inner = sympify(func_str)
                    backend_used = "sympify_original"
                    result["notes"].append({"level": "info", "msg": "sympify succeeded on original inner function string"})
                except Exception as e2:
                    result["notes"].append({"level": "error", "msg": f"Failed to parse inner function: {e2}"})
                    result["sympy"] = None
                    result["backend"] = backend_used
                    return serialize_result(result)

        result["sympy"] = inner
        result["backend"] = backend_used

        # compute derivative
        try:
            var_sym = sp.Symbol(var)
            if compute_derivative:
                derived = sp.diff(inner, var_sym, order)
                result["derived"] = sp.simplify(derived)
                result["notes"].append({"level": "info", "msg": "Computed derivative successfully"})
            else:
                # store unsimplified derivative object (or None)
                result["derived"] = sp.diff(inner, var_sym, order)
                result["notes"].append({"level": "info", "msg": "Parsed derivative but did not compute (compute_derivative=False)"})
        except Exception as e:
            result["notes"].append({"level": "error", "msg": f"Derivative computation failed: {e}"})

        return serialize_result(result)

    # Not a derivative pattern -> try regular expression parsing
    result["kind"] = "expr"
    backend_used = None

    # Heuristic: treat as latex if backslash or braces or leading $
    is_latex = ("\\" in expr) or ("{" in expr) or ("}" in expr) or expr.strip().startswith("$")
    if mode_hint == "latex":
        is_latex = True

    # 1) parse_latex
    if is_latex and parse_latex is not None:
        try:
            parsed = parse_latex(expr)
            result["sympy"] = parsed
            backend_used = "parse_latex"
            result["notes"].append({"level": "info", "msg": "parse_latex succeeded"})
            result["backend"] = backend_used
            return serialize_result(result)
        except Exception as e:
            result["notes"].append({"level": "info", "msg": f"parse_latex failed: {e}"})

    # 2) latex2sympy2
    if latex2sympy2_func is not None:
        parsed, notes = try_latex2sympy2(expr)
        result["notes"].extend(notes)
        if parsed is not None:
            result["sympy"] = parsed
            backend_used = "latex2sympy2"
            result["backend"] = backend_used
            result["notes"].append({"level": "info", "msg": "latex2sympy2 succeeded"})
            return serialize_result(result)

    # 3) fallback: robust ascii conversion and sympify
    try:
        ascii_expr = expr if not is_latex else latex_to_ascii_robust(expr)
        ascii_expr = ascii_expr.strip()
        result["ascii_fallback"] = ascii_expr
        parsed = sympify(ascii_expr)
        result["sympy"] = parsed
        backend_used = "sympify_ascii"
        result["notes"].append({"level": "info", "msg": "sympify succeeded on ascii-fallback"})
        result["backend"] = backend_used
        return serialize_result(result)
    except Exception as e:
        result["notes"].append({"level": "warn", "msg": f"sympify on ascii-fallback failed: {e}"})
        # last attempt: sympify original
        try:
            parsed = sympify(expr)
            result["sympy"] = parsed
            backend_used = "sympify_original"
            result["backend"] = backend_used
            result["notes"].append({"level": "info", "msg": "sympify succeeded on original expr"})
            return serialize_result(result)
        except Exception as e2:
            result["notes"].append({"level": "error", "msg": f"Failed to parse expression. ascii-fallback error: {e}; sympify(original) error: {e2}"})
            result["sympy"] = None
            result["backend"] = backend_used
            return serialize_result(result)


if __name__ == "__main__":
    examples = [
        r"x^2 + 3*x - 4",
        r"\frac{d}{dx} x^2",
        r"\sin(x) + 2",
        r"\frac{1}{2} x^2 + 3x + 1",
        r"$\frac{a}{b} + c$",
        r"d/dx (x**3 + 2*x)",
        r"\frac{d}{dx} \sin(x)",
        r"\frac{\frac{1}{2}x^2 + 1}{x+1}",
        r"\frac{d}{dx} \frac{1}{2} x^2",
        # some tricky cases
        r"\frac{1+\frac{a}{b}}{c}",
        r"2x + 3(x+1)",
        r"\frac{d^2}{dx^2} x^4",  # second derivative
    ]

    for ex in examples:
        print("\n" + "=" * 80)
        print("INPUT :", ex)
        try:
            out = parse_expression(ex)
            # pretty-print some important fields
            print("KIND  :", out["kind"])
            print("BACKEND:", out.get("backend"))
            print("SUCCESS:", out.get("success"))
            if out.get("ascii_fallback"):
                print("ASCII fallback:", out["ascii_fallback"])
            print("SYMpy :", out.get("sympy_str"))
            if out.get("derived_str") is not None:
                print("DERIV :", out.get("derived_str"))
            print("FREE_SYMS:", out.get("free_symbols"))
            print("CANONICAL:", out.get("canonical_str"))
            print("AST   :", out.get("ast"))
            # show notes concisely
            notes = out.get("notes", [])
            print("NOTES :")
            for n in notes:
                level = n.get("level", "info")
                msg = n.get("msg", "")
                print(f"  - [{level}] {msg}")
        except Exception as err:
            print("ERROR :", err)
    print("\nDone.")


def test_derivative_with_python_exponent():
    out = parse_expression("d/dx (x**3 + 2*x)")
    assert out["success"]
    assert out["sympy_str"] == "x**3 + 2*x"
    assert out["derived_str"] == "3*x**2 + 2"

def test_implicit_multiplication():
    out = parse_expression("2x + 3(x+1)")
    assert out["success"]
    assert out["sympy_str"] == "2*x + 3*(x + 1)" or "2*x + 3*(x + 1)" in out["sympy_str"]

def test_nested_fraction():
    out = parse_expression(r"\frac{1+\frac{a}{b}}{c}")
    assert out["success"]
    assert out["sympy_str"] in {"(a/b + 1)/c", "(1 + a/b)/c"}

def test_second_derivative():
    out = parse_expression(r"\frac{d^2}{dx^2} x^4")
    assert out["success"]
    assert out["derived_str"] == "12*x**2"
