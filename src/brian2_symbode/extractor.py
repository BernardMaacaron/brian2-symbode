"""
extractor.py
============
Extracts symbolic ODE systems from Brian2 Network objects.

Brian2 stores neuron dynamics as string-based equations and distributes synaptic
coupling across separate Synapse objects via implicit "summed" variables.  This
module reconstructs the full symbolic ODE for each neuron group by:

1. **Recursive variable resolution** — tracing variables through NeuronGroup
   equations, Synapse summed_updaters, and pre/post references.
2. **Structured decomposition** — separating element-wise local dynamics from
   matrix-vector synaptic coupling terms (needed for correct vectorised
   simulation with PyTorch, JAX, etc.).
3. **Function mapping** — replacing opaque Brian2 functions (e.g. custom C++/CUDA
   transfer functions) with SymPy-compatible symbolic equivalents.

Key entry points
----------------
- ``SymbolicGraphExtractor``  — main class
- ``get_derived_ode()``       — flat symbolic ODE (good for Jacobians)
- ``compute_full_jacobian()`` — system-level Jacobian matrix
"""

import sympy as sp
from brian2 import Synapses
from brian2.parsing.sympytools import str_to_sympy


class SymbolicGraphExtractor:
    """Extract symbolic ODE systems from a Brian2 Network.

    Handles the implicit "summed" variable mechanism that Brian2 uses for
    synaptic coupling.  Variables like ``I_EE`` appear in NeuronGroup equations
    as plain parameters, but their values are actually computed by Synapse
    objects and written via ``summed_updaters``.  This class traces those
    connections and builds a complete symbolic representation.

    Two extraction modes are available:

    - **Flat** (``get_derived_ode``): Returns a single monolithic SymPy
      expression with all variables inlined.  Useful for Jacobians and
      analytical stability analysis.
    - **Structured** (``get_structured_ode``): Returns a dict separating
      local (element-wise) dynamics from coupling (matrix-vector) terms,
      with full parameter metadata.  Useful for building differentiable
      simulations in PyTorch / JAX.

    Parameters
    ----------
    brian_objects : list or brian2.Network
        The Brian2 objects (NeuronGroups, Synapses) to analyse.  Accepts
        either a ``Network`` object (reads ``.objects``) or a plain list.
    function_map : dict, optional
        Mapping from Brian2 function names (str) to SymPy-compatible
        callables.  Each callable must accept and return SymPy expressions.
        Example::

            {'f': f_symbolic}
            {'clip': lambda x, a, b: sp.Max(a, sp.Min(x, b))}

    verbose : bool, optional
        If True, prints a detailed trace of every resolution step.
        Useful for debugging equation extraction.

    Attributes
    ----------
    objects : list
        All Brian2 objects being analysed.
    function_map : dict
        The active function-name → callable mapping.
    symbol_cache : dict
        Cache of previously resolved symbols (currently unused, reserved
        for future memoisation).
    object_names : dict
        Quick lookup from ``object.name`` → object.

    Examples
    --------
    >>> from brian2_symbode import SymbolicGraphExtractor, get_derived_ode
    >>> ext = SymbolicGraphExtractor(net, function_map={'f': f_symbolic})
    >>> ode = get_derived_ode(ext, exc_group, 'x')       # flat expression
    >>> structured = ext.get_structured_ode(exc_group, 'x')  # structured dict
    """

    def __init__(self, brian_objects, function_map=None, verbose=False):
        # Accept either a Network (has .objects attribute) or a raw list
        if hasattr(brian_objects, 'objects'):
            self.objects = list(brian_objects.objects)
        else:
            self.objects = list(brian_objects)

        self.function_map = function_map or {}
        self.verbose = verbose
        self.symbol_cache = {}                                      # Reserved for future memoisation
        self.object_names = {obj.name: obj for obj in self.objects}  # name → object lookup

    def log(self, message, depth=0):
        """Print an indented debug message when ``verbose=True``.

        Parameters
        ----------
        message : str
            The message to print.
        depth : int
            Indentation level (each level = 2 spaces).
        """
        if self.verbose:
            indent = "  " * depth
            print(f"[brian2-symbode] {indent}{message}")

    # ------------------------------------------------------------------
    # Core recursive resolution
    # ------------------------------------------------------------------

    def resolve_variable(self, variable_name, owner_object, visited=None, depth=0):
        """Recursively resolve a variable to its fundamental symbolic expression.

        This is the core algorithm of the extractor.  It follows a strict
        priority order:

        1. **Summed variable** — If any Synapse writes a ``(summed)``
           variable to ``owner_object`` matching ``variable_name``, trace
           into that synapse's expression and resolve recursively.
        2. **Equation lookup** — If the variable is in the owner's
           ``equations`` dict:

           - *Differential equation* → leaf symbol (state variable).
           - *Constant parameter* → leaf symbol.
           - *Subexpression* → parse RHS and recurse.
           - *Non-constant parameter* → leaf symbol.

        3. **Synapse pre/post references** — In a ``Synapses`` object,
           ``var_pre`` jumps to the source group and ``var_post`` jumps
           to the target group.
        4. **Fallback** → return an opaque ``sp.Symbol``.

        Parameters
        ----------
        variable_name : str
            The variable to resolve (e.g. ``'I_EE'``, ``'r'``, ``'x'``).
        owner_object : brian2 object
            The NeuronGroup or Synapses that "owns" this variable.
        visited : set, optional
            Tracks ``(variable_name, object_name)`` pairs already visited
            to prevent infinite recursion from circular references.
        depth : int
            Current recursion depth (used for indented logging).

        Returns
        -------
        sympy.Expr
            The fully resolved symbolic expression.  Leaf variables are
            returned as ``sp.Symbol('varname_objectname')``.

        Notes
        -----
        The ``visited`` set is **copied** before descending into
        ``_resolve_expression_symbols`` so that sibling symbols in
        the same expression don't block each other.
        """
        if visited is None:
            visited = set()

        unique_id = (variable_name, owner_object.name)
        if unique_id in visited:
            self.log(f"(!) Cycle/Cache hit for {variable_name} in {owner_object.name}", depth)
            return sp.Symbol(f"{variable_name}_{owner_object.name}")

        visited.add(unique_id)
        self.log(f"Resolving '{variable_name}' in object '{owner_object.name}'...", depth)

        # 1. Check if it's a summed variable written by an incoming synapse.
        #    This is crucial: variables like 'I_EE' don't exist in the neuron
        #    equations as subexpressions — they are injected by synapses.
        incoming_synapses = self._find_targeting_synapses(owner_object, variable_name)
        if incoming_synapses:
            self.log(
                f"-> Found {len(incoming_synapses)} incoming synapse(s) "
                f"writing to '{variable_name}'", depth + 1,
            )
            accumulated_expr = 0
            for syn, expr_str in incoming_synapses:
                self.log(
                    f"-> Tracing synapse '{syn.name}' with expression: {expr_str}",
                    depth + 2,
                )
                synapse_expr = self._resolve_expression_symbols(
                    str_to_sympy(expr_str), syn, visited, depth + 3,
                )
                accumulated_expr += synapse_expr
            return accumulated_expr

        # 2. Check if it's a standard equation (diff eq, subexpression, or parameter)
        if hasattr(owner_object, 'equations') and variable_name in owner_object.equations:
            eq = owner_object.equations[variable_name]

            if eq.type == 'differential equation':
                self.log("-> Is Differential Equation state variable. Stop recursion.", depth + 1)
                return sp.Symbol(f"{variable_name}_{owner_object.name}")

            if eq.type in ['subexpression', 'parameter']:
                if "constant" in eq.flags:
                    self.log("-> Is Constant Parameter.", depth + 1)
                    return sp.Symbol(f"{variable_name}_{owner_object.name}")

                if hasattr(eq, 'expr') and eq.expr:
                    self.log(f"-> Is Subexpression: {eq.expr}", depth + 1)
                    if hasattr(eq.expr, 'code'):
                        rhs_sym = str_to_sympy(eq.expr.code)
                    else:
                        rhs_sym = str_to_sympy(str(eq.expr))
                    return self._resolve_expression_symbols(
                        rhs_sym, owner_object, visited, depth + 1,
                    )
                else:
                    self.log("-> Is Parameter without expression.", depth + 1)
                    return sp.Symbol(f"{variable_name}_{owner_object.name}")

        # 3. Check for external references in Synapses (e.g. '_pre', '_post')
        if isinstance(owner_object, Synapses):
            if variable_name.endswith('_pre'):
                target_var = variable_name[:-4]
                self.log(
                    f"-> Jumping to PRE-synaptic group "
                    f"'{owner_object.source.name}' for '{target_var}'", depth + 1,
                )
                return self.resolve_variable(target_var, owner_object.source, visited, depth + 1)
            elif variable_name.endswith('_post'):
                target_var = variable_name[:-5]
                self.log(
                    f"-> Jumping to POST-synaptic group "
                    f"'{owner_object.target.name}' for '{target_var}'", depth + 1,
                )
                return self.resolve_variable(target_var, owner_object.target, visited, depth + 1)

        # 4. Fallback (basic symbol)
        self.log("-> Unresolved/Leaf variable. Returning symbol.", depth + 1)
        return sp.Symbol(f"{variable_name}_{owner_object.name}")

    # ------------------------------------------------------------------
    # Expression & function helpers
    # ------------------------------------------------------------------

    def _resolve_expression_symbols(self, sympy_expr, owner, visited, depth=0):
        """Resolve every free symbol in a SymPy expression via :meth:`resolve_variable`.

        Iterates over ``sympy_expr.free_symbols``, resolves each one
        recursively, substitutes the results back, and then applies
        ``function_map`` to expand any custom functions.

        Parameters
        ----------
        sympy_expr : sympy.Expr
            A SymPy expression whose free symbols need resolution.
        owner : brian2 object
            The NeuronGroup or Synapses that owns this expression.
        visited : set
            The cycle-detection set from the calling ``resolve_variable``.
            Each symbol gets a **copy** so siblings don't block each other.
        depth : int
            Current recursion depth for logging.

        Returns
        -------
        sympy.Expr
            The expression with all symbols fully resolved.
        """
        if visited is None:
            visited = set()

        subs_dict = {}
        # Walk every free symbol (atom) in the expression
        for atom in sympy_expr.free_symbols:
            if isinstance(atom, sp.Function):
                continue  # Functions are handled by _replace_functions later

            atom_name = str(atom)
            # Resolve this symbol recursively; use a COPY of visited so
            # sibling atoms remain reachable (they haven't been visited yet)
            resolved = self.resolve_variable(atom_name, owner, visited.copy(), depth)
            subs_dict[atom] = resolved

        # Perform all substitutions in one pass
        resolved_expr = sympy_expr.subs(subs_dict)

        # Replace custom functions (e.g. f → Piecewise) if a map is provided
        if self.function_map:
            resolved_expr = self._replace_functions(resolved_expr)

        return resolved_expr

    def _replace_functions(self, expr):
        """Recursively replace functions defined in ``function_map``.

        Walks the SymPy expression tree.  When a ``Function`` node is found
        whose name exists in ``self.function_map``, it is replaced by
        calling the mapped callable with the (recursively processed) arguments.

        Parameters
        ----------
        expr : sympy.Expr
            The expression to transform.

        Returns
        -------
        sympy.Expr
            The expression with all mapped functions expanded.
        """
        if isinstance(expr, sp.Function):
            func_name = expr.func.__name__
            new_args = [self._replace_functions(arg) for arg in expr.args]

            if func_name in self.function_map:
                return self.function_map[func_name](*new_args)
            return expr.func(*new_args)

        # Traverse children for composite objects (Add, Mul, Pow, etc.)
        if isinstance(expr, sp.Basic) and expr.args:
            new_args = [self._replace_functions(arg) for arg in expr.args]
            if new_args != list(expr.args):
                return expr.func(*new_args)

        return expr

    # ------------------------------------------------------------------
    # Discretisation
    # ------------------------------------------------------------------

    def discretise(self, ode_expr, state_var, integration_method, dt):
        """Convert a continuous ODE into a discrete update map.

        Converts ``dx/dt = f(x)`` into ``x_{n+1} = G(x_n)`` using the
        specified numerical integration method.

        Parameters
        ----------
        ode_expr : sympy.Expr
            The right-hand side of the ODE, f(x).
        state_var : sympy.Symbol
            The state variable being evolved.
        integration_method : str
            One of ``'euler'``, ``'rk2'`` (or ``'heun'``),
            ``'exponential_euler'``.
        dt : sympy.Symbol or float
            The simulation timestep.

        Returns
        -------
        sympy.Expr
            The symbolic expression for the updated state ``x_new``.

        Raises
        ------
        ValueError
            If ``integration_method`` is not recognised.
        """
        method = integration_method.lower()

        if method == 'euler':
            # Forward Euler: x_new = x + dt * f(x)
            return state_var + dt * ode_expr

        elif method in ('rk2', 'heun'):
            # Midpoint method (RK2):
            # k1 = f(x);  k2 = f(x + dt/2 * k1);  x_new = x + dt * k2
            k1 = ode_expr
            k2 = ode_expr.subs(state_var, state_var + (dt / 2) * k1)
            return state_var + dt * k2

        elif method == 'exponential_euler':
            # For linear ODEs dx/dt = -Ax + B:
            # x_new = x * exp(-A*dt) + (B/A) * (1 - exp(-A*dt))
            linear_coeff = sp.diff(ode_expr, state_var)

            if linear_coeff.has(state_var):
                print(
                    f"[Warning] Exponential Euler requested for non-linear ODE "
                    f"'{ode_expr}'. Falling back to Euler."
                )
                return self.discretise(ode_expr, state_var, 'euler', dt)

            A_decay = -linear_coeff
            B_input = ode_expr.subs(state_var, 0)

            if A_decay == 0:
                return self.discretise(ode_expr, state_var, 'euler', dt)

            decay_factor = sp.exp(-A_decay * dt)
            return state_var * decay_factor + (B_input / A_decay) * (1 - decay_factor)

        else:
            raise ValueError(f"Unknown integration method: {integration_method}")

    # ------------------------------------------------------------------
    # Internal: synapse discovery
    # ------------------------------------------------------------------

    def _find_targeting_synapses(self, target_group, var_name):
        """Find all Synapse objects that write a summed variable into *target_group*.

        Brian2 summed variables use the convention ``I_EE_post`` inside the
        Synapse namespace to write into ``I_EE`` on the post-synaptic
        NeuronGroup.  This method checks both naming conventions.

        Parameters
        ----------
        target_group : brian2.NeuronGroup
            The post-synaptic group to search for.
        var_name : str
            The variable name as it appears in the NeuronGroup equations
            (e.g. ``'I_EE'``).

        Returns
        -------
        list of (Synapses, str)
            Each tuple contains the Synapse object and the expression
            string from its ``summed_updaters``.
            Empty list if no synapse targets this group with this variable.
        """
        hits = []
        for obj in self.objects:
            if isinstance(obj, Synapses):
                if obj.target == target_group:
                    candidate_names = [f"{var_name}_post", var_name]
                    for cand in candidate_names:
                        if hasattr(obj, 'summed_updaters') and cand in obj.summed_updaters:
                            updater = obj.summed_updaters[cand]
                            hits.append((obj, str(updater.expression)))
                            break  # Found the match for this synapse
        return hits

    def _is_summed_target_anywhere(self, var_name):
        """Check whether *var_name* is a summed-variable target in ANY synapse.

        Used for **dead variable detection**: if a variable is a summed target
        somewhere (returns True) but ``_find_targeting_synapses`` returns
        nothing for a specific group, then that variable is "dead" (always 0)
        for that group.

        Parameters
        ----------
        var_name : str
            The variable name to check (e.g. ``'I_EI'``).

        Returns
        -------
        bool
            True if any Synapse object has a summed_updater for this variable.
        """
        for obj in self.objects:
            if isinstance(obj, Synapses):
                candidate_names = [f"{var_name}_post", var_name]
                for cand in candidate_names:
                    if hasattr(obj, 'summed_updaters') and cand in obj.summed_updaters:
                        return True
        return False

    # ------------------------------------------------------------------
    # Structured ODE extraction
    # ------------------------------------------------------------------

    def get_structured_ode(self, group, state_var):
        """Decompose a group's ODE into local dynamics and coupling terms.

        Unlike ``get_derived_ode()`` which returns a flat symbolic expression,
        this method separates element-wise operations from synaptic coupling
        that requires matrix-vector products.

        Parameters
        ----------
        group : brian2.NeuronGroup
            The neuron group to extract the ODE for.
        state_var : str
            Name of the state variable (e.g. ``'x'``, ``'v'``).

        Returns
        -------
        dict
            Keys:

            ``'local_expr'`` : sympy.Expr
                Element-wise local dynamics with all summed variables
                (live and dead) replaced with 0.
            ``'coupling_terms'`` : list of dict
                One entry per incoming synapse, each with keys
                ``'synapse'``, ``'weight_symbol'``, ``'source_group'``,
                ``'activation_expr'``, ``'coupling_coeff_expr'``.
                ``coupling_coeff_expr`` is the symbolic multiplicative factor
                that scales this summed input in the ODE, i.e. dRHS/dI_sum.
            ``'dead_symbols'`` : list of sympy.Symbol
                Symbols that are structurally zero (summed targets with
                no incoming synapse for this group).
            ``'state_symbol'`` : sympy.Symbol
                The state symbol for this group.
            ``'free_params'`` : list of sympy.Symbol
                All non-state parameter symbols.
            ``'param_metadata'`` : dict
                Per-parameter metadata from Brian2 (source object, flags,
                sizing, etc.).
        """
        from brian2 import NeuronGroup as BNeuronGroup
        from brian2.parsing.sympytools import str_to_sympy as _str_to_sympy

        eq = group.equations[state_var]
        rhs_str = eq.expr.code
        rhs_sym = _str_to_sympy(rhs_str)

        state_sym = sp.Symbol(f"{state_var}_{group.name}")

        # --- Three-way classification of each atom in the RHS ---
        #
        # 1. COUPLING: A summed variable for this group → decompose into
        #    weight + activation for matrix-vector product.
        # 2. DEAD: A summed variable somewhere else, but NOT for this group
        #    → structurally zero.
        # 3. LOCAL: A regular variable → resolve normally.
        #
        coupling_terms = []
        dead_symbols = []
        subs_dict = {}

        for atom in rhs_sym.free_symbols:
            atom_name = str(atom)
            incoming = self._find_targeting_synapses(group, atom_name)

            if incoming:
                # COUPLING
                # Coupling coefficient in RHS for this summed variable.
                # Example SOC: RHS = (-x + I_EE + I_IE)/tau -> coeff = 1/tau.
                coeff_expr = sp.diff(rhs_sym, atom)
                if coeff_expr.has(atom):
                    raise ValueError(
                        f"Non-linear dependence on summed variable '{atom_name}' "
                        f"in group '{group.name}' is not supported by "
                        "structured coupling decomposition."
                    )
                coeff_expr = self._resolve_expression_symbols(
                    coeff_expr, group, visited=None, depth=0,
                )
                if self.function_map:
                    coeff_expr = self._replace_functions(coeff_expr)

                for syn, expr_str in incoming:
                    syn_expr = _str_to_sympy(expr_str)
                    weight_sym, activation_expr = self._decompose_synapse_expr(
                        syn_expr, syn, atom_name,
                    )
                    coupling_terms.append({
                        'synapse': syn,
                        'weight_symbol': weight_sym,
                        'source_group': syn.source,
                        'activation_expr': activation_expr,
                        'coupling_coeff_expr': coeff_expr,
                    })
                subs_dict[atom] = sp.Integer(0)

            elif self._is_summed_target_anywhere(atom_name):
                # DEAD — summed variable with no synapse targeting this group
                dead_sym = sp.Symbol(f"{atom_name}_{group.name}")
                dead_symbols.append(dead_sym)
                subs_dict[atom] = sp.Integer(0)

            else:
                # LOCAL — resolve via recursive algorithm
                resolved = self.resolve_variable(atom_name, group, visited=None, depth=0)
                subs_dict[atom] = resolved

        # Build local expression with couplings and dead vars removed
        local_expr = rhs_sym.subs(subs_dict)

        if self.function_map:
            local_expr = self._replace_functions(local_expr)

        # --- Collect free parameter symbols (excluding state variables) ---
        all_free = set(local_expr.free_symbols)
        for ct in coupling_terms:
            all_free.update(ct['activation_expr'].free_symbols)
            all_free.update(ct['coupling_coeff_expr'].free_symbols)
            all_free.add(ct['weight_symbol'])

        # Separate state variables from parameters
        state_names = {state_sym.name}
        free_params = []
        for s in sorted(all_free, key=lambda s: s.name):
            if s.name in state_names:
                continue
            # Check if this symbol is the state var of another NeuronGroup
            is_other_state = False
            for obj in self.objects:
                if isinstance(obj, BNeuronGroup) and obj != group:
                    candidate = f"{state_var}_{obj.name}"
                    if s.name == candidate:
                        is_other_state = True
                        state_names.add(s.name)
                        break
            if not is_other_state:
                free_params.append(s)

        # --- Build parameter metadata ---
        param_metadata = {}
        for sym in free_params:
            meta = self._get_param_metadata(sym, group)
            param_metadata[sym.name] = meta

        for ct in coupling_terms:
            ws = ct['weight_symbol']
            if ws.name not in param_metadata:
                meta = self._get_param_metadata(ws, group)
                param_metadata[ws.name] = meta

        return {
            'local_expr': local_expr,
            'coupling_terms': coupling_terms,
            'dead_symbols': dead_symbols,
            'state_symbol': state_sym,
            'free_params': free_params,
            'param_metadata': param_metadata,
        }

    # ------------------------------------------------------------------
    # Internal: synapse expression decomposition
    # ------------------------------------------------------------------

    def _decompose_synapse_expr(self, syn_expr, synapse_obj, target_var_name):
        """Separate a synapse summed expression into weight and activation.

        Given a synapse expression like ``we * r_pre``, this method:

        1. Resolves all symbols via ``resolve_variable``.
        2. Identifies the **weight** — the symbol that is a ``parameter``
           in the synapse's own equations.
        3. Divides out the weight to obtain the **activation**.

        Parameters
        ----------
        syn_expr : sympy.Expr
            Parsed expression from the synapse's summed_updater.
        synapse_obj : brian2.Synapses
            The Synapse object that owns this expression.
        target_var_name : str
            Name of the summed variable being written (for logging).

        Returns
        -------
        weight_symbol : sympy.Symbol or sympy.Integer
            The resolved weight symbol, or ``sp.Integer(1)`` if none found.
        activation_expr : sympy.Expr
            The fully resolved pre-synaptic activation expression.
        """
        # Step 1: Resolve every atom
        visited = set()
        resolved_parts = {}

        for atom in syn_expr.free_symbols:
            atom_name = str(atom)
            resolved = self.resolve_variable(atom_name, synapse_obj, visited.copy(), depth=0)
            resolved_parts[atom_name] = resolved

        # Step 2: Classify each atom as weight or activation
        weight_sym = None
        activation_atoms = {}

        for atom in syn_expr.free_symbols:
            atom_name = str(atom)
            resolved = resolved_parts[atom_name]

            if hasattr(synapse_obj, 'equations') and atom_name in synapse_obj.equations:
                eq = synapse_obj.equations[atom_name]
                if eq.type == 'parameter':
                    weight_sym = resolved
                    continue

            activation_atoms[atom] = resolved

        # Step 3: Extract activation by dividing out the weight
        activation_expr = syn_expr

        if weight_sym is not None:
            weight_atom = None
            for atom in syn_expr.free_symbols:
                if resolved_parts[str(atom)] == weight_sym:
                    weight_atom = atom
                    break
            if weight_atom is not None:
                activation_expr = sp.cancel(syn_expr / weight_atom)

        # Step 4: Resolve remaining symbols in activation
        act_subs = {}
        for atom in activation_expr.free_symbols:
            atom_name = str(atom)
            if atom_name in resolved_parts:
                act_subs[atom] = resolved_parts[atom_name]

        activation_expr = activation_expr.subs(act_subs)

        if self.function_map:
            activation_expr = self._replace_functions(activation_expr)

        # Fallback: no weight found → unit weight
        if weight_sym is None:
            weight_sym = sp.Integer(1)
            activation_expr = syn_expr.subs(
                {a: resolved_parts[str(a)] for a in syn_expr.free_symbols}
            )
            if self.function_map:
                activation_expr = self._replace_functions(activation_expr)

        return weight_sym, activation_expr

    # ------------------------------------------------------------------
    # Internal: parameter metadata
    # ------------------------------------------------------------------

    def _get_param_metadata(self, symbol, reference_group):
        """Extract Brian2 metadata for a parameter symbol.

        Reverse-maps a SymPy symbol name (e.g. ``g_exc_neurons``) back to
        the Brian2 object that owns it.

        Parameters
        ----------
        symbol : sympy.Symbol
            The symbol to look up.
        reference_group : brian2.NeuronGroup
            Fallback context (not the primary lookup mechanism).

        Returns
        -------
        dict
            Keys: ``source_obj``, ``var_name``, ``flags``, ``eq_type``,
            ``is_per_neuron``, ``group_size``.
        """
        from brian2 import NeuronGroup as BNeuronGroup

        name = symbol.name

        # Match by suffix: sort by name length descending to avoid false matches
        objs = sorted(self.objects, key=lambda o: len(o.name), reverse=True)

        for obj in objs:
            if name.endswith(f"_{obj.name}"):
                var_name = name[:-(len(obj.name) + 1)]

                flags = set()
                eq_type = 'unknown'
                is_per_neuron = True
                group_size = 1

                if hasattr(obj, 'equations') and var_name in obj.equations:
                    eq = obj.equations[var_name]
                    eq_type = eq.type
                    flags = set(eq.flags) if hasattr(eq, 'flags') else set()

                    if 'shared' in flags:
                        is_per_neuron = False

                if isinstance(obj, BNeuronGroup):
                    group_size = len(obj)
                elif isinstance(obj, Synapses):
                    group_size = (len(obj.target), len(obj.source))
                    is_per_neuron = False

                return {
                    'source_obj': obj,
                    'var_name': var_name,
                    'flags': flags,
                    'eq_type': eq_type,
                    'is_per_neuron': is_per_neuron,
                    'group_size': group_size,
                }

        return {
            'source_obj': None,
            'var_name': name,
            'flags': set(),
            'eq_type': 'unknown',
            'is_per_neuron': True,
            'group_size': 1,
        }


# ======================================================================
# Module-level convenience functions
# ======================================================================

def get_derived_ode(extractor, group, state_var):
    """Get the fully resolved flat ODE for a state variable.

    Extracts the RHS of ``d(state_var)/dt`` as a single monolithic SymPy
    expression.  All summed variables are inlined, all subexpressions
    expanded, and the ``function_map`` is applied.

    .. note::
       Dead summed variables are NOT detected in flat mode — they remain
       as opaque symbols.  Use ``extractor.get_structured_ode()`` for
       clean dead-variable handling.

    Parameters
    ----------
    extractor : SymbolicGraphExtractor
        A configured extractor instance.
    group : brian2.NeuronGroup
        The neuron group to extract the ODE for.
    state_var : str
        Name of the state variable (e.g. ``'x'``).

    Returns
    -------
    sympy.Expr
        The RHS of ``d(state_var)/dt`` as a fully resolved expression.
    """
    if extractor.verbose:
        print(f"\n[brian2-symbode] Starting extraction for {group.name} [{state_var}]")

    eq = group.equations[state_var]
    rhs_str = eq.expr.code
    rhs_sym = str_to_sympy(rhs_str)
    return extractor._resolve_expression_symbols(rhs_sym, group, visited=None, depth=0)


def compute_full_jacobian(ode_expressions, state_symbols):
    r"""Compute the symbolic Jacobian matrix of an ODE system.

    .. math::
        J_{ij} = \frac{\partial F_i}{\partial x_j}

    Parameters
    ----------
    ode_expressions : list of sympy.Expr
        RHS expressions ``[dx1/dt, dx2/dt, ...]``.
    state_symbols : list of sympy.Symbol
        State variables ``[x1, x2, ...]`` in matching order.

    Returns
    -------
    sympy.Matrix
        Jacobian of shape ``(len(ode_expressions), len(state_symbols))``.
    """
    return sp.Matrix([
        [sp.diff(ode, var) for var in state_symbols]
        for ode in ode_expressions
    ])
