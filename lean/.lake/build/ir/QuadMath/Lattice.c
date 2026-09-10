// Lean compiler output
// Module: QuadMath.Lattice
// Imports: public import Init public meta import Init public import QuadMath.IVM
#include <lean/lean.h>
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wunused-label"
#elif defined(__GNUC__) && !defined(__CLANG__)
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-label"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif
#ifdef __cplusplus
extern "C" {
#endif
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_List_range(lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Quadlean_omniNumber(lean_object*);
LEAN_EXPORT lean_object* lp_Quadlean_omniNumber___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_Quadlean_List_mapTR_loop___at___00cumulativeCount_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Quadlean_List_foldr___at___00List_sum___at___00cumulativeCount_spec__1_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Quadlean_List_foldr___at___00List_sum___at___00cumulativeCount_spec__1_spec__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Quadlean_List_sum___at___00cumulativeCount_spec__1(lean_object*);
LEAN_EXPORT lean_object* lp_Quadlean_List_sum___at___00cumulativeCount_spec__1___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_Quadlean_cumulativeCount(lean_object*);
LEAN_EXPORT lean_object* lp_Quadlean_omniNumber(lean_object* v_k_1_){
_start:
{
lean_object* v___x_2_; lean_object* v___x_3_; lean_object* v___x_4_; lean_object* v___x_5_; lean_object* v___x_6_; 
v___x_2_ = lean_unsigned_to_nat(10u);
v___x_3_ = lean_nat_mul(v___x_2_, v_k_1_);
v___x_4_ = lean_nat_mul(v___x_3_, v_k_1_);
lean_dec(v___x_3_);
v___x_5_ = lean_unsigned_to_nat(2u);
v___x_6_ = lean_nat_add(v___x_4_, v___x_5_);
lean_dec(v___x_4_);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_Quadlean_omniNumber___boxed(lean_object* v_k_7_){
_start:
{
lean_object* v_res_8_; 
v_res_8_ = lp_Quadlean_omniNumber(v_k_7_);
lean_dec(v_k_7_);
return v_res_8_;
}
}
LEAN_EXPORT lean_object* lp_Quadlean_List_mapTR_loop___at___00cumulativeCount_spec__0(lean_object* v_a_9_, lean_object* v_a_10_){
_start:
{
if (lean_obj_tag(v_a_9_) == 0)
{
lean_object* v___x_11_; 
v___x_11_ = l_List_reverse___redArg(v_a_10_);
return v___x_11_;
}
else
{
lean_object* v_head_12_; lean_object* v_tail_13_; lean_object* v___x_15_; uint8_t v_isShared_16_; uint8_t v_isSharedCheck_24_; 
v_head_12_ = lean_ctor_get(v_a_9_, 0);
v_tail_13_ = lean_ctor_get(v_a_9_, 1);
v_isSharedCheck_24_ = !lean_is_exclusive(v_a_9_);
if (v_isSharedCheck_24_ == 0)
{
v___x_15_ = v_a_9_;
v_isShared_16_ = v_isSharedCheck_24_;
goto v_resetjp_14_;
}
else
{
lean_inc(v_tail_13_);
lean_inc(v_head_12_);
lean_dec(v_a_9_);
v___x_15_ = lean_box(0);
v_isShared_16_ = v_isSharedCheck_24_;
goto v_resetjp_14_;
}
v_resetjp_14_:
{
lean_object* v___x_17_; lean_object* v___x_18_; lean_object* v___x_19_; lean_object* v___x_21_; 
v___x_17_ = lean_unsigned_to_nat(1u);
v___x_18_ = lean_nat_add(v_head_12_, v___x_17_);
lean_dec(v_head_12_);
v___x_19_ = lp_Quadlean_omniNumber(v___x_18_);
lean_dec(v___x_18_);
if (v_isShared_16_ == 0)
{
lean_ctor_set(v___x_15_, 1, v_a_10_);
lean_ctor_set(v___x_15_, 0, v___x_19_);
v___x_21_ = v___x_15_;
goto v_reusejp_20_;
}
else
{
lean_object* v_reuseFailAlloc_23_; 
v_reuseFailAlloc_23_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_23_, 0, v___x_19_);
lean_ctor_set(v_reuseFailAlloc_23_, 1, v_a_10_);
v___x_21_ = v_reuseFailAlloc_23_;
goto v_reusejp_20_;
}
v_reusejp_20_:
{
v_a_9_ = v_tail_13_;
v_a_10_ = v___x_21_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Quadlean_List_foldr___at___00List_sum___at___00cumulativeCount_spec__1_spec__1(lean_object* v_init_25_, lean_object* v_x_26_){
_start:
{
if (lean_obj_tag(v_x_26_) == 0)
{
lean_inc(v_init_25_);
return v_init_25_;
}
else
{
lean_object* v_head_27_; lean_object* v_tail_28_; lean_object* v___x_29_; lean_object* v___x_30_; 
v_head_27_ = lean_ctor_get(v_x_26_, 0);
v_tail_28_ = lean_ctor_get(v_x_26_, 1);
v___x_29_ = lp_Quadlean_List_foldr___at___00List_sum___at___00cumulativeCount_spec__1_spec__1(v_init_25_, v_tail_28_);
v___x_30_ = lean_nat_add(v_head_27_, v___x_29_);
lean_dec(v___x_29_);
return v___x_30_;
}
}
}
LEAN_EXPORT lean_object* lp_Quadlean_List_foldr___at___00List_sum___at___00cumulativeCount_spec__1_spec__1___boxed(lean_object* v_init_31_, lean_object* v_x_32_){
_start:
{
lean_object* v_res_33_; 
v_res_33_ = lp_Quadlean_List_foldr___at___00List_sum___at___00cumulativeCount_spec__1_spec__1(v_init_31_, v_x_32_);
lean_dec(v_x_32_);
lean_dec(v_init_31_);
return v_res_33_;
}
}
LEAN_EXPORT lean_object* lp_Quadlean_List_sum___at___00cumulativeCount_spec__1(lean_object* v_l_34_){
_start:
{
lean_object* v___x_35_; lean_object* v___x_36_; 
v___x_35_ = lean_unsigned_to_nat(0u);
v___x_36_ = lp_Quadlean_List_foldr___at___00List_sum___at___00cumulativeCount_spec__1_spec__1(v___x_35_, v_l_34_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_Quadlean_List_sum___at___00cumulativeCount_spec__1___boxed(lean_object* v_l_37_){
_start:
{
lean_object* v_res_38_; 
v_res_38_ = lp_Quadlean_List_sum___at___00cumulativeCount_spec__1(v_l_37_);
lean_dec(v_l_37_);
return v_res_38_;
}
}
LEAN_EXPORT lean_object* lp_Quadlean_cumulativeCount(lean_object* v_k_39_){
_start:
{
lean_object* v___x_40_; lean_object* v___x_41_; lean_object* v___x_42_; lean_object* v___x_43_; lean_object* v___x_44_; lean_object* v___x_45_; 
v___x_40_ = lean_unsigned_to_nat(1u);
v___x_41_ = l_List_range(v_k_39_);
v___x_42_ = lean_box(0);
v___x_43_ = lp_Quadlean_List_mapTR_loop___at___00cumulativeCount_spec__0(v___x_41_, v___x_42_);
v___x_44_ = lp_Quadlean_List_sum___at___00cumulativeCount_spec__1(v___x_43_);
lean_dec(v___x_43_);
v___x_45_ = lean_nat_add(v___x_40_, v___x_44_);
lean_dec(v___x_44_);
return v___x_45_;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Quadlean_QuadMath_IVM(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_Quadlean_QuadMath_Lattice(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Quadlean_QuadMath_IVM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
