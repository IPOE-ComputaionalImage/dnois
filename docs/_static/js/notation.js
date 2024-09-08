function registerMacro(name, definition) {
    if (!('MathJax' in window)) {
        window.MathJax = {};
    }
    if (!('tex' in window.MathJax)) {
        window.MathJax.tex = {};
    }
    if (!('macros' in window.MathJax.tex)) {
        window.MathJax.tex.macros = {};
    }
    window.MathJax.tex.macros[name] = definition;
}

registerMacro('d', '\\text{d}');
registerMacro('e', '\\text{e}');
registerMacro('ft', '\\mathcal{F}');
registerMacro('i', '\\text{i}');
registerMacro('pfrac', ['\\frac{\\partial {#1}}{\\partial {#2}}', 2])

registerMacro('rect', '\\text{rect}');
registerMacro('sinc', '\\text{sinc}');
registerMacro('sgn', '\\text{sgn}');
registerMacro('trg', '\\Lambda');
registerMacro('comb', '\\text{comb}');
registerMacro('circfunc', '\\text{circ}');
