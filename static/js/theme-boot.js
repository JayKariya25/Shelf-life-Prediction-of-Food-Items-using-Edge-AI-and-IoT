/* Runs synchronously in <head> before first paint so a dark-mode user never
   sees a white flash. Kept as its own file because the CSP forbids inline JS. */
(function () {
    try {
        var stored = localStorage.getItem("slp.theme");
        if (stored === "dark" || stored === "light") {
            document.documentElement.setAttribute("data-theme", stored);
        }
    } catch (error) {
        /* storage blocked: fall back to the prefers-color-scheme media query */
    }
})();
