import sympy as sp
import re

# Allow exp and log during sympify
SAFE_GLOBALS = {
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
}

# ================================================================
# Utility: Preprocess expression text
# ================================================================
def preprocess(expr: str) -> str:
    expr = expr.strip()
    expr = expr.replace("^", "**")
    expr = expr.replace("=", "==")
    expr = re.sub(r'(?<=\d)(?=[A-Za-z(])', '*', expr)
    expr = re.sub(r'(?<=[A-Za-z])(?=\()', '*', expr)
    expr = re.sub(r'(?<=\))(?=\()', ')*(', expr)  # handle (x+1)(x-1)
    expr = re.sub(r'(?<=\))(?=[A-Za-z0-9])', ')*', expr)
    return expr


# ================================================================
# Equation Type Detection
# ================================================================
def detect_equation_type(expr):
    try:
        if not isinstance(expr, sp.Eq):
            return "EXPRESSION"

        lhs, rhs = expr.args
        diff_expr = sp.simplify(lhs - rhs)

        variables = diff_expr.free_symbols
        if not variables:
            return "CONSTANT"

        max_degree = max(sp.degree(diff_expr, var) for var in variables)

        expr_str = str(diff_expr)
        if "exp" in expr_str:
            return "EXPONENTIAL"
        if "log" in expr_str:
            return "LOGARITHMIC"
        if max_degree == 1:
            return "LINEAR"
        elif max_degree == 2:
            return "QUADRATIC"
        elif max_degree == 3:
            return "CUBIC"
        else:
            return "POLYNOMIAL"
    except Exception:
        return "UNKNOWN"


# ================================================================
# Recursively convert expression to template
# ================================================================
def build_template(expr):
    if isinstance(expr, sp.Eq):
        left, right = expr.args
        return f"subtract({build_template(left)},{build_template(right)})"
    elif isinstance(expr, sp.Add):
        parts = [build_template(a) for a in expr.args]
        return f"add({','.join(parts)})"
    elif isinstance(expr, sp.Mul):
        parts = [build_template(a) for a in expr.args]
        return f"multiply({','.join(parts)})"
    elif isinstance(expr, sp.Pow):
        base, exp = expr.args
        return f"power({build_template(base)},{build_template(exp)})"
    elif isinstance(expr, sp.Function):
        func_name = expr.func.__name__
        args = [build_template(a) for a in expr.args]
        return f"{func_name}({','.join(args)})"
    elif isinstance(expr, sp.Symbol):
        return f"ARG_{expr}"
    elif isinstance(expr, (sp.Integer, sp.Float)):
        return f"CONST_{expr}"
    else:
        return str(expr)


# ================================================================
# Extract arguments (symbols and constants)
# ================================================================
def extract_arguments(expr):
    symbols = sorted(list(expr.free_symbols), key=lambda s: s.name)
    args = {f"ARG{i}": str(s) for i, s in enumerate(symbols)}
    consts = {}
    i = 1
    for num in expr.atoms(sp.Number):
        consts[f"CONST_{i}"] = str(num)
        i += 1
    return args, consts


# ================================================================
# Main entry point
# ================================================================
def expression_to_template(input_expr: str):
    result = {"original": input_expr}
    try:
        clean = preprocess(input_expr)
        expr = sp.sympify(clean, evaluate=False, locals=SAFE_GLOBALS)

        eq_type = detect_equation_type(expr)
        template_str = build_template(expr)
        args, consts = extract_arguments(expr)
        arg_text = ", ".join([f"{k}: {v}" for k, v in {**args, **consts}.items()])

        formatted = (
            f"Template: {eq_type}:{template_str}. "
            f"Arguments: {arg_text}. "
            f"Task: Generate a realistic math word problem that matches this equation."
        )

        result.update({
            "success": True,
            "type": eq_type,
            "template": template_str,
            "arguments": arg_text,
            "formatted_input": formatted
        })

    except Exception as e:
        result.update({"success": False, "error": str(e)})

    return result


# ================================================================
# Test
# ================================================================
if __name__ == "__main__":
    samples = [
        "2x + 3 = 9",
        "x^2 + 3x + 2 = 0",
        "a^3 + 2a^2 + a + 1 = 0",
        "5 * exp(x) = 20",
        "log(x) + 3 = 5",
        "(x+1)(x-1)"
    ]

    for s in samples:
        print("=" * 80)
        parsed = expression_to_template(s)
        print("Input:", s)
        if parsed["success"]:
            print("Type:", parsed["type"])
            print("Template:", parsed["template"])
            print("Arguments:", parsed["arguments"])
            print("Formatted Input:\n", parsed["formatted_input"])
        else:
            print("Error:", parsed["error"])
