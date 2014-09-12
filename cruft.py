# Various partials and cross-partials of intensive form of F
        ftheta            = sp.diff(self.intensive_output(x, y, theta), theta, 1)
        fthetatheta       = sp.diff(self.intensive_output(x, y, theta), theta, 2)
        fxy               = sp.diff(self.intensive_output(x, y, theta), x, 1, 
                                    y, 1)
        fytheta           = sp.diff(self.intensive_output(x, y, theta), y, 1, 
                                    theta, 1)
        fxtheta           = sp.diff(self.intensive_output(x, y, theta), x, 1, 
                                    theta, 1)
        fx                = sp.diff(self.intensive_output(x, y, theta), x, 1)
        fxx               = sp.diff(self.intensive_output(x, y, theta), x, 2)
        fxthetatheta      = sp.diff(self.intensive_output(x, y, theta), y, 1, 
                                    theta, 2)
        fythetatheta      = sp.diff(self.intensive_output(x, y, theta), y, 1, 
                                    theta, 2)
        fthetathetatheta  = sp.diff(self.intensive_output(x, y, theta), theta, 3)
        
        # store all of the derivatives in a dict
        derivatives_dict = {'Fx':Fx, 'Fy':Fy, 'Fl':Fl, 'Fr':Fr, 'Fxy':Fxy, 
                            'Fyl':Fyl, 'Fxr':Fxr, 'Fyl':Fyl, 'Flr':Flr, 
                            'Fyly':Fyly, 'Fxry':Fxry, 'Flry':Flry,
                            'ftheta':ftheta, 'fthetatheta':fthetatheta, 
                            'fxy':fxy, 'fytheta':fytheta, 'fxtheta':fxtheta, 
                            'fx':fx, 'fxx':fxx, 'fxthetatheta':fxthetatheta,
                            'fythetatheta':fythetatheta, 
                            'fthetathetatheta':fthetathetatheta}

    def get_jacobian(self, xn, vec, matching='positive'):
        """
        Jacobian for the positive/negative assortative matching system.
        Providing the Jacobian of the system as an input to the ODE solver can 
        significantly increase the computational efficiency of the forward 
        shooting algorithm. 

        Arguments:

            vec:      Vector of variables, vec = [theta, mu]
            xn:       Worker type/skill.
            matching: One of either 'positive' or 'negative', depending on 
                      whether you are computing a PAM or NAM equilibrium.

        Returns: A list-of-lists defining the Jacobian for the system of 
        differential equations.

        TODO: Use numerical differentiation to compute (d h_f / d mu) for generic
        h_f. Currently only handles uniform h_f (in which case this derivative is
        zero!).
        
        """
        # values defining the putative equilibrium path
        vals = {x:xn, y:vec[1], l:vec[0], r:1.0, theta:vec[0]}
            
        # [0,0] entry for the jacobian requires the following derivatives
        Fyl      = self.derivatives['Fyl'].evalf(subs=vals)
        Fxr      = self.derivatives['Fxr'].evalf(subs=vals)
        Flr      = self.derivatives['Flr'].evalf(subs=vals)
        Fyltheta = self.derivatives['fythetatheta'].evalf(subs=vals)
        Fxrtheta = (self.derivatives['fxtheta'].evalf(subs=vals) - vec[0] *
                    self.derivatives['fxthetatheta'].evalf(subs=vals))
        Flrtheta = -(self.derivatives['fthetatheta'].evalf(subs=vals) + 
                     self.derivatives['fthetathetatheta'].evalf(subs=vals))

        # [0,1] entry for the jacobian requires the following derivatives
        Hy   = -self.H(xn, vec) * (0.0 / self.firmProdPDF(vec[1])) * (1 / vec[0]) 
        Fyly = self.derivatives['Fyly'].evalf(subs=vals)
        Fxry = self.derivatives['Fxry'].evalf(subs=vals) 
        Flry = self.derivatives['Flry'].evalf(subs=vals) 
        
        # build the jacobian
        jac00 = ((Flr * (self.H(xn, vec) * Fyltheta -  Fxrtheta) - 
                  (self.H(xn, vec) * Fyl -  Fxr) * Flrtheta) / Flr**2)
        jac01 = ((Flr * (Hy * Fyl + self.H(xn, vec) * Fyly - Fxry) - 
                  (self.H(xn, vec) * Fyl - Fxr) * Flry) / Flr**2)
        jac10 = -(self.H(xn, vec) / vec[0]**2)
        jac11 = Hy 

        if matching == 'positive':
            return [[jac00, jac01], [jac10, jac11]]
        elif matching == 'negative':
            return [[-jac00, -jac01], [-jac10, -jac11]]
        else:
            raise Exception, ("Matching must be one of either 'positive' or " +
                              "'negative'.")
