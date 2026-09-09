# Authentication vs authorization

Authentication answers who the caller is (password, OTP, OAuth token,
mTLS). Authorization answers what they may do (roles, permissions,
resource ownership). Mixing them produces the 401/403 confusion: 401 is
authn failure; 403 is authz failure after a successful authn.

Implement authn at the edge (verify JWT signature and expiry). Implement
authz in the application next to the resource (`order.user_id == token.sub`).
