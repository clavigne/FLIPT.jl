E1 = 3.94
E2 = 4.89
# 4 D model of pyrazine (data from pyr4op)
H = ( 
    # HO part
    Dict("10a" => " 0.1139 * W"),
    Dict( "6a" => " 0.0739 * W"),
    Dict( "el" => "$E1 * S(2,2) + $E2 * S(3,3)"),

    # linear, on-diagonal coupling coefficients
    # H(1,1)
    Dict("6a"  => "-0.09806 * q", "el"=>"S(2,2)"), 

    # H(2,2)
    Dict("6a"  => " 0.13545 * q", "el"=>"S(3,3)"),

    # quadratic, on-diagonal coupling coefficients
    # Dict("10a" => "-0.01159 * q * q", "el"=>"S(2,2)+S(3,3)"),

    # linear, off-diagonal coupling coefficients
    Dict("10a"  => " 0.183 * q", "el" => "S(2,3)"))


# Regexp to convert between different numbers of el surfaces 
# S(\([0-9]\),\([0-9]\))
# S(\,(+ (string-to-number \1) -1),\,(+ (string-to-number \2) -1))
