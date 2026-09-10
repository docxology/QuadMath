// Lean compiler output
// Module: QuadMath.Quadray
// Imports: public import Init public meta import Init
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
lean_object* lean_nat_to_int(lean_object*);
uint8_t lean_int_dec_le(lean_object*, lean_object*);
lean_object* lean_string_length(lean_object*);
uint8_t lean_int_dec_lt(lean_object*, lean_object*);
lean_object* l_Int_repr(lean_object*);
lean_object* l_Repr_addAppParen(lean_object*, lean_object*);
lean_object* lean_int_mul(lean_object*, lean_object*);
lean_object* lean_int_sub(lean_object*, lean_object*);
lean_object* lean_int_add(lean_object*, lean_object*);
uint8_t lean_int_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_abs(lean_object*);
lean_object* l_mkRat(lean_object*, lean_object*);
static const lean_string_object lp_Quadlean_instReprQuadray_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "{ "};
static const lean_object* lp_Quadlean_instReprQuadray_repr___redArg___closed__0 = (const lean_object*)&lp_Quadlean_instReprQuadray_repr___redArg___closed__0_value;
static const lean_string_object lp_Quadlean_instReprQuadray_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "a"};
static const lean_object* lp_Quadlean_instReprQuadray_repr___redArg___closed__1 = (const lean_object*)&lp_Quadlean_instReprQuadray_repr___redArg___closed__1_value;
static const lean_ctor_object lp_Quadlean_instReprQuadray_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_Quadlean_instReprQuadray_repr___redArg___closed__1_value)}};
static const lean_object* lp_Quadlean_instReprQuadray_repr___redArg___closed__2 = (const lean_object*)&lp_Quadlean_instReprQuadray_repr___redArg___closed__2_value;
static const lean_ctor_object lp_Quadlean_instReprQuadray_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Quadlean_instReprQuadray_repr___redArg___closed__2_value)}};
static const lean_object* lp_Quadlean_instReprQuadray_repr___redArg___closed__3 = (const lean_object*)&lp_Quadlean_instReprQuadray_repr___redArg___closed__3_value;
static const lean_string_object lp_Quadlean_instReprQuadray_repr___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " := "};
static const lean_object* lp_Quadlean_instReprQuadray_repr___redArg___closed__4 = (const lean_object*)&lp_Quadlean_instReprQuadray_repr___redArg___closed__4_value;
static const lean_ctor_object lp_Quadlean_instReprQuadray_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_Quadlean_instReprQuadray_repr___redArg___closed__4_value)}};
static const lean_object* lp_Quadlean_instReprQuadray_repr___redArg___closed__5 = (const lean_object*)&lp_Quadlean_instReprQuadray_repr___redArg___closed__5_value;
static const lean_ctor_object lp_Quadlean_instReprQuadray_repr___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_Quadlean_instReprQuadray_repr___redArg___closed__3_value),((lean_object*)&lp_Quadlean_instReprQuadray_repr___redArg___closed__5_value)}};
static const lean_object* lp_Quadlean_instReprQuadray_repr___redArg___closed__6 = (const lean_object*)&lp_Quadlean_instReprQuadray_repr___redArg___closed__6_value;
static lean_once_cell_t lp_Quadlean_instReprQuadray_repr___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Quadlean_instReprQuadray_repr___redArg___closed__7;
static const lean_string_object lp_Quadlean_instReprQuadray_repr___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " }"};
static const lean_object* lp_Quadlean_instReprQuadray_repr___redArg___closed__8 = (const lean_object*)&lp_Quadlean_instReprQuadray_repr___redArg___closed__8_value;
static lean_once_cell_t lp_Quadlean_instReprQuadray_repr___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Quadlean_instReprQuadray_repr___redArg___closed__9;
static lean_once_cell_t lp_Quadlean_instReprQuadray_repr___redArg___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Quadlean_instReprQuadray_repr___redArg___closed__10;
static const lean_ctor_object lp_Quadlean_instReprQuadray_repr___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_Quadlean_instReprQuadray_repr___redArg___closed__0_value)}};
static const lean_object* lp_Quadlean_instReprQuadray_repr___redArg___closed__11 = (const lean_object*)&lp_Quadlean_instReprQuadray_repr___redArg___closed__11_value;
static const lean_ctor_object lp_Quadlean_instReprQuadray_repr___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_Quadlean_instReprQuadray_repr___redArg___closed__8_value)}};
static const lean_object* lp_Quadlean_instReprQuadray_repr___redArg___closed__12 = (const lean_object*)&lp_Quadlean_instReprQuadray_repr___redArg___closed__12_value;
static const lean_string_object lp_Quadlean_instReprQuadray_repr___redArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "d"};
static const lean_object* lp_Quadlean_instReprQuadray_repr___redArg___closed__13 = (const lean_object*)&lp_Quadlean_instReprQuadray_repr___redArg___closed__13_value;
static const lean_ctor_object lp_Quadlean_instReprQuadray_repr___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_Quadlean_instReprQuadray_repr___redArg___closed__13_value)}};
static const lean_object* lp_Quadlean_instReprQuadray_repr___redArg___closed__14 = (const lean_object*)&lp_Quadlean_instReprQuadray_repr___redArg___closed__14_value;
static lean_once_cell_t lp_Quadlean_instReprQuadray_repr___redArg___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Quadlean_instReprQuadray_repr___redArg___closed__15;
static const lean_string_object lp_Quadlean_instReprQuadray_repr___redArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "c"};
static const lean_object* lp_Quadlean_instReprQuadray_repr___redArg___closed__16 = (const lean_object*)&lp_Quadlean_instReprQuadray_repr___redArg___closed__16_value;
static const lean_ctor_object lp_Quadlean_instReprQuadray_repr___redArg___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_Quadlean_instReprQuadray_repr___redArg___closed__16_value)}};
static const lean_object* lp_Quadlean_instReprQuadray_repr___redArg___closed__17 = (const lean_object*)&lp_Quadlean_instReprQuadray_repr___redArg___closed__17_value;
static const lean_string_object lp_Quadlean_instReprQuadray_repr___redArg___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_Quadlean_instReprQuadray_repr___redArg___closed__18 = (const lean_object*)&lp_Quadlean_instReprQuadray_repr___redArg___closed__18_value;
static const lean_ctor_object lp_Quadlean_instReprQuadray_repr___redArg___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_Quadlean_instReprQuadray_repr___redArg___closed__18_value)}};
static const lean_object* lp_Quadlean_instReprQuadray_repr___redArg___closed__19 = (const lean_object*)&lp_Quadlean_instReprQuadray_repr___redArg___closed__19_value;
static const lean_string_object lp_Quadlean_instReprQuadray_repr___redArg___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "b"};
static const lean_object* lp_Quadlean_instReprQuadray_repr___redArg___closed__20 = (const lean_object*)&lp_Quadlean_instReprQuadray_repr___redArg___closed__20_value;
static const lean_ctor_object lp_Quadlean_instReprQuadray_repr___redArg___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_Quadlean_instReprQuadray_repr___redArg___closed__20_value)}};
static const lean_object* lp_Quadlean_instReprQuadray_repr___redArg___closed__21 = (const lean_object*)&lp_Quadlean_instReprQuadray_repr___redArg___closed__21_value;
LEAN_EXPORT lean_object* lp_Quadlean_instReprQuadray_repr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_Quadlean_instReprQuadray_repr___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_Quadlean_instReprQuadray_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Quadlean_instReprQuadray_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_Quadlean_instReprQuadray___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_Quadlean_instReprQuadray_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_Quadlean_instReprQuadray___closed__0 = (const lean_object*)&lp_Quadlean_instReprQuadray___closed__0_value;
LEAN_EXPORT const lean_object* lp_Quadlean_instReprQuadray = (const lean_object*)&lp_Quadlean_instReprQuadray___closed__0_value;
LEAN_EXPORT uint8_t lp_Quadlean_instDecidableEqQuadray_decEq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Quadlean_instDecidableEqQuadray_decEq___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_Quadlean_instDecidableEqQuadray(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Quadlean_instDecidableEqQuadray___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Quadlean_Quadray_add(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Quadlean_Quadray_add___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Quadlean_Quadray_sub(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Quadlean_Quadray_sub___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Quadlean_quadMin(lean_object*);
LEAN_EXPORT lean_object* lp_Quadlean_quadMin___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_Quadlean_Quadray_normalize(lean_object*);
LEAN_EXPORT lean_object* lp_Quadlean_proj(lean_object*);
LEAN_EXPORT lean_object* lp_Quadlean_proj___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_Quadlean_det3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Quadlean_det3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Quadlean_tetraDet(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Quadlean_tetraDet___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Quadlean_tetraVolume(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Quadlean_tetraVolume___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_Quadlean_instReprQuadray_repr___redArg___closed__7(void){
_start:
{
lean_object* v___x_14_; lean_object* v___x_15_; 
v___x_14_ = lean_unsigned_to_nat(5u);
v___x_15_ = lean_nat_to_int(v___x_14_);
return v___x_15_;
}
}
static lean_object* _init_lp_Quadlean_instReprQuadray_repr___redArg___closed__9(void){
_start:
{
lean_object* v___x_17_; lean_object* v___x_18_; 
v___x_17_ = ((lean_object*)(lp_Quadlean_instReprQuadray_repr___redArg___closed__0));
v___x_18_ = lean_string_length(v___x_17_);
return v___x_18_;
}
}
static lean_object* _init_lp_Quadlean_instReprQuadray_repr___redArg___closed__10(void){
_start:
{
lean_object* v___x_19_; lean_object* v___x_20_; 
v___x_19_ = lean_obj_once(&lp_Quadlean_instReprQuadray_repr___redArg___closed__9, &lp_Quadlean_instReprQuadray_repr___redArg___closed__9_once, _init_lp_Quadlean_instReprQuadray_repr___redArg___closed__9);
v___x_20_ = lean_nat_to_int(v___x_19_);
return v___x_20_;
}
}
static lean_object* _init_lp_Quadlean_instReprQuadray_repr___redArg___closed__15(void){
_start:
{
lean_object* v___x_28_; lean_object* v___x_29_; 
v___x_28_ = lean_unsigned_to_nat(0u);
v___x_29_ = lean_nat_to_int(v___x_28_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_Quadlean_instReprQuadray_repr___redArg(lean_object* v_x_39_){
_start:
{
lean_object* v_a_40_; lean_object* v_b_41_; lean_object* v_c_42_; lean_object* v_d_43_; lean_object* v___x_44_; lean_object* v___x_45_; lean_object* v___x_46_; lean_object* v___y_48_; uint8_t v___y_49_; lean_object* v___y_50_; lean_object* v___y_62_; uint8_t v___y_63_; lean_object* v___y_64_; lean_object* v___y_65_; lean_object* v___y_66_; lean_object* v___y_84_; lean_object* v___y_85_; uint8_t v___y_86_; lean_object* v___y_87_; lean_object* v___y_88_; lean_object* v___y_106_; lean_object* v___x_126_; lean_object* v___x_127_; uint8_t v___x_128_; 
v_a_40_ = lean_ctor_get(v_x_39_, 0);
v_b_41_ = lean_ctor_get(v_x_39_, 1);
v_c_42_ = lean_ctor_get(v_x_39_, 2);
v_d_43_ = lean_ctor_get(v_x_39_, 3);
v___x_44_ = ((lean_object*)(lp_Quadlean_instReprQuadray_repr___redArg___closed__5));
v___x_45_ = ((lean_object*)(lp_Quadlean_instReprQuadray_repr___redArg___closed__6));
v___x_46_ = lean_obj_once(&lp_Quadlean_instReprQuadray_repr___redArg___closed__7, &lp_Quadlean_instReprQuadray_repr___redArg___closed__7_once, _init_lp_Quadlean_instReprQuadray_repr___redArg___closed__7);
v___x_126_ = lean_unsigned_to_nat(0u);
v___x_127_ = lean_obj_once(&lp_Quadlean_instReprQuadray_repr___redArg___closed__15, &lp_Quadlean_instReprQuadray_repr___redArg___closed__15_once, _init_lp_Quadlean_instReprQuadray_repr___redArg___closed__15);
v___x_128_ = lean_int_dec_lt(v_a_40_, v___x_127_);
if (v___x_128_ == 0)
{
lean_object* v___x_129_; lean_object* v___x_130_; 
v___x_129_ = l_Int_repr(v_a_40_);
v___x_130_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_130_, 0, v___x_129_);
v___y_106_ = v___x_130_;
goto v___jp_105_;
}
else
{
lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v___x_133_; 
v___x_131_ = l_Int_repr(v_a_40_);
v___x_132_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_132_, 0, v___x_131_);
v___x_133_ = l_Repr_addAppParen(v___x_132_, v___x_126_);
v___y_106_ = v___x_133_;
goto v___jp_105_;
}
v___jp_47_:
{
lean_object* v___x_51_; lean_object* v___x_52_; lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v___x_59_; lean_object* v___x_60_; 
v___x_51_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_51_, 0, v___x_46_);
lean_ctor_set(v___x_51_, 1, v___y_50_);
v___x_52_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_52_, 0, v___x_51_);
lean_ctor_set_uint8(v___x_52_, sizeof(void*)*1, v___y_49_);
v___x_53_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_53_, 0, v___y_48_);
lean_ctor_set(v___x_53_, 1, v___x_52_);
v___x_54_ = lean_obj_once(&lp_Quadlean_instReprQuadray_repr___redArg___closed__10, &lp_Quadlean_instReprQuadray_repr___redArg___closed__10_once, _init_lp_Quadlean_instReprQuadray_repr___redArg___closed__10);
v___x_55_ = ((lean_object*)(lp_Quadlean_instReprQuadray_repr___redArg___closed__11));
v___x_56_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_56_, 0, v___x_55_);
lean_ctor_set(v___x_56_, 1, v___x_53_);
v___x_57_ = ((lean_object*)(lp_Quadlean_instReprQuadray_repr___redArg___closed__12));
v___x_58_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_58_, 0, v___x_56_);
lean_ctor_set(v___x_58_, 1, v___x_57_);
v___x_59_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_59_, 0, v___x_54_);
lean_ctor_set(v___x_59_, 1, v___x_58_);
v___x_60_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_60_, 0, v___x_59_);
lean_ctor_set_uint8(v___x_60_, sizeof(void*)*1, v___y_49_);
return v___x_60_;
}
v___jp_61_:
{
lean_object* v___x_67_; lean_object* v___x_68_; lean_object* v___x_69_; lean_object* v___x_70_; lean_object* v___x_71_; lean_object* v___x_72_; lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; lean_object* v___x_76_; uint8_t v___x_77_; 
v___x_67_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_67_, 0, v___x_46_);
lean_ctor_set(v___x_67_, 1, v___y_66_);
v___x_68_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_68_, 0, v___x_67_);
lean_ctor_set_uint8(v___x_68_, sizeof(void*)*1, v___y_63_);
v___x_69_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_69_, 0, v___y_65_);
lean_ctor_set(v___x_69_, 1, v___x_68_);
lean_inc(v___y_62_);
v___x_70_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_70_, 0, v___x_69_);
lean_ctor_set(v___x_70_, 1, v___y_62_);
lean_inc(v___y_64_);
v___x_71_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_71_, 0, v___x_70_);
lean_ctor_set(v___x_71_, 1, v___y_64_);
v___x_72_ = ((lean_object*)(lp_Quadlean_instReprQuadray_repr___redArg___closed__14));
v___x_73_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_73_, 0, v___x_71_);
lean_ctor_set(v___x_73_, 1, v___x_72_);
v___x_74_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_74_, 0, v___x_73_);
lean_ctor_set(v___x_74_, 1, v___x_44_);
v___x_75_ = lean_unsigned_to_nat(0u);
v___x_76_ = lean_obj_once(&lp_Quadlean_instReprQuadray_repr___redArg___closed__15, &lp_Quadlean_instReprQuadray_repr___redArg___closed__15_once, _init_lp_Quadlean_instReprQuadray_repr___redArg___closed__15);
v___x_77_ = lean_int_dec_lt(v_d_43_, v___x_76_);
if (v___x_77_ == 0)
{
lean_object* v___x_78_; lean_object* v___x_79_; 
v___x_78_ = l_Int_repr(v_d_43_);
v___x_79_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_79_, 0, v___x_78_);
v___y_48_ = v___x_74_;
v___y_49_ = v___y_63_;
v___y_50_ = v___x_79_;
goto v___jp_47_;
}
else
{
lean_object* v___x_80_; lean_object* v___x_81_; lean_object* v___x_82_; 
v___x_80_ = l_Int_repr(v_d_43_);
v___x_81_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_81_, 0, v___x_80_);
v___x_82_ = l_Repr_addAppParen(v___x_81_, v___x_75_);
v___y_48_ = v___x_74_;
v___y_49_ = v___y_63_;
v___y_50_ = v___x_82_;
goto v___jp_47_;
}
}
v___jp_83_:
{
lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; uint8_t v___x_99_; 
v___x_89_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_89_, 0, v___x_46_);
lean_ctor_set(v___x_89_, 1, v___y_88_);
v___x_90_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_90_, 0, v___x_89_);
lean_ctor_set_uint8(v___x_90_, sizeof(void*)*1, v___y_86_);
v___x_91_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_91_, 0, v___y_84_);
lean_ctor_set(v___x_91_, 1, v___x_90_);
lean_inc(v___y_85_);
v___x_92_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_92_, 0, v___x_91_);
lean_ctor_set(v___x_92_, 1, v___y_85_);
lean_inc(v___y_87_);
v___x_93_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_93_, 0, v___x_92_);
lean_ctor_set(v___x_93_, 1, v___y_87_);
v___x_94_ = ((lean_object*)(lp_Quadlean_instReprQuadray_repr___redArg___closed__17));
v___x_95_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_95_, 0, v___x_93_);
lean_ctor_set(v___x_95_, 1, v___x_94_);
v___x_96_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_96_, 0, v___x_95_);
lean_ctor_set(v___x_96_, 1, v___x_44_);
v___x_97_ = lean_unsigned_to_nat(0u);
v___x_98_ = lean_obj_once(&lp_Quadlean_instReprQuadray_repr___redArg___closed__15, &lp_Quadlean_instReprQuadray_repr___redArg___closed__15_once, _init_lp_Quadlean_instReprQuadray_repr___redArg___closed__15);
v___x_99_ = lean_int_dec_lt(v_c_42_, v___x_98_);
if (v___x_99_ == 0)
{
lean_object* v___x_100_; lean_object* v___x_101_; 
v___x_100_ = l_Int_repr(v_c_42_);
v___x_101_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_101_, 0, v___x_100_);
v___y_62_ = v___y_85_;
v___y_63_ = v___y_86_;
v___y_64_ = v___y_87_;
v___y_65_ = v___x_96_;
v___y_66_ = v___x_101_;
goto v___jp_61_;
}
else
{
lean_object* v___x_102_; lean_object* v___x_103_; lean_object* v___x_104_; 
v___x_102_ = l_Int_repr(v_c_42_);
v___x_103_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_103_, 0, v___x_102_);
v___x_104_ = l_Repr_addAppParen(v___x_103_, v___x_97_);
v___y_62_ = v___y_85_;
v___y_63_ = v___y_86_;
v___y_64_ = v___y_87_;
v___y_65_ = v___x_96_;
v___y_66_ = v___x_104_;
goto v___jp_61_;
}
}
v___jp_105_:
{
lean_object* v___x_107_; uint8_t v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; lean_object* v___x_116_; lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; uint8_t v___x_120_; 
v___x_107_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_107_, 0, v___x_46_);
lean_ctor_set(v___x_107_, 1, v___y_106_);
v___x_108_ = 0;
v___x_109_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_109_, 0, v___x_107_);
lean_ctor_set_uint8(v___x_109_, sizeof(void*)*1, v___x_108_);
v___x_110_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_110_, 0, v___x_45_);
lean_ctor_set(v___x_110_, 1, v___x_109_);
v___x_111_ = ((lean_object*)(lp_Quadlean_instReprQuadray_repr___redArg___closed__19));
v___x_112_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_112_, 0, v___x_110_);
lean_ctor_set(v___x_112_, 1, v___x_111_);
v___x_113_ = lean_box(1);
v___x_114_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_114_, 0, v___x_112_);
lean_ctor_set(v___x_114_, 1, v___x_113_);
v___x_115_ = ((lean_object*)(lp_Quadlean_instReprQuadray_repr___redArg___closed__21));
v___x_116_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_116_, 0, v___x_114_);
lean_ctor_set(v___x_116_, 1, v___x_115_);
v___x_117_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_117_, 0, v___x_116_);
lean_ctor_set(v___x_117_, 1, v___x_44_);
v___x_118_ = lean_unsigned_to_nat(0u);
v___x_119_ = lean_obj_once(&lp_Quadlean_instReprQuadray_repr___redArg___closed__15, &lp_Quadlean_instReprQuadray_repr___redArg___closed__15_once, _init_lp_Quadlean_instReprQuadray_repr___redArg___closed__15);
v___x_120_ = lean_int_dec_lt(v_b_41_, v___x_119_);
if (v___x_120_ == 0)
{
lean_object* v___x_121_; lean_object* v___x_122_; 
v___x_121_ = l_Int_repr(v_b_41_);
v___x_122_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_122_, 0, v___x_121_);
v___y_84_ = v___x_117_;
v___y_85_ = v___x_111_;
v___y_86_ = v___x_108_;
v___y_87_ = v___x_113_;
v___y_88_ = v___x_122_;
goto v___jp_83_;
}
else
{
lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; 
v___x_123_ = l_Int_repr(v_b_41_);
v___x_124_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_124_, 0, v___x_123_);
v___x_125_ = l_Repr_addAppParen(v___x_124_, v___x_118_);
v___y_84_ = v___x_117_;
v___y_85_ = v___x_111_;
v___y_86_ = v___x_108_;
v___y_87_ = v___x_113_;
v___y_88_ = v___x_125_;
goto v___jp_83_;
}
}
}
}
LEAN_EXPORT lean_object* lp_Quadlean_instReprQuadray_repr___redArg___boxed(lean_object* v_x_134_){
_start:
{
lean_object* v_res_135_; 
v_res_135_ = lp_Quadlean_instReprQuadray_repr___redArg(v_x_134_);
lean_dec_ref(v_x_134_);
return v_res_135_;
}
}
LEAN_EXPORT lean_object* lp_Quadlean_instReprQuadray_repr(lean_object* v_x_136_, lean_object* v_prec_137_){
_start:
{
lean_object* v___x_138_; 
v___x_138_ = lp_Quadlean_instReprQuadray_repr___redArg(v_x_136_);
return v___x_138_;
}
}
LEAN_EXPORT lean_object* lp_Quadlean_instReprQuadray_repr___boxed(lean_object* v_x_139_, lean_object* v_prec_140_){
_start:
{
lean_object* v_res_141_; 
v_res_141_ = lp_Quadlean_instReprQuadray_repr(v_x_139_, v_prec_140_);
lean_dec(v_prec_140_);
lean_dec_ref(v_x_139_);
return v_res_141_;
}
}
LEAN_EXPORT uint8_t lp_Quadlean_instDecidableEqQuadray_decEq(lean_object* v_x_144_, lean_object* v_x_145_){
_start:
{
lean_object* v_a_146_; lean_object* v_b_147_; lean_object* v_c_148_; lean_object* v_d_149_; lean_object* v_a_150_; lean_object* v_b_151_; lean_object* v_c_152_; lean_object* v_d_153_; uint8_t v___x_154_; 
v_a_146_ = lean_ctor_get(v_x_144_, 0);
v_b_147_ = lean_ctor_get(v_x_144_, 1);
v_c_148_ = lean_ctor_get(v_x_144_, 2);
v_d_149_ = lean_ctor_get(v_x_144_, 3);
v_a_150_ = lean_ctor_get(v_x_145_, 0);
v_b_151_ = lean_ctor_get(v_x_145_, 1);
v_c_152_ = lean_ctor_get(v_x_145_, 2);
v_d_153_ = lean_ctor_get(v_x_145_, 3);
v___x_154_ = lean_int_dec_eq(v_a_146_, v_a_150_);
if (v___x_154_ == 0)
{
return v___x_154_;
}
else
{
uint8_t v___x_155_; 
v___x_155_ = lean_int_dec_eq(v_b_147_, v_b_151_);
if (v___x_155_ == 0)
{
return v___x_155_;
}
else
{
uint8_t v___x_156_; 
v___x_156_ = lean_int_dec_eq(v_c_148_, v_c_152_);
if (v___x_156_ == 0)
{
return v___x_156_;
}
else
{
uint8_t v___x_157_; 
v___x_157_ = lean_int_dec_eq(v_d_149_, v_d_153_);
return v___x_157_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Quadlean_instDecidableEqQuadray_decEq___boxed(lean_object* v_x_158_, lean_object* v_x_159_){
_start:
{
uint8_t v_res_160_; lean_object* v_r_161_; 
v_res_160_ = lp_Quadlean_instDecidableEqQuadray_decEq(v_x_158_, v_x_159_);
lean_dec_ref(v_x_159_);
lean_dec_ref(v_x_158_);
v_r_161_ = lean_box(v_res_160_);
return v_r_161_;
}
}
LEAN_EXPORT uint8_t lp_Quadlean_instDecidableEqQuadray(lean_object* v_x_162_, lean_object* v_x_163_){
_start:
{
uint8_t v___x_164_; 
v___x_164_ = lp_Quadlean_instDecidableEqQuadray_decEq(v_x_162_, v_x_163_);
return v___x_164_;
}
}
LEAN_EXPORT lean_object* lp_Quadlean_instDecidableEqQuadray___boxed(lean_object* v_x_165_, lean_object* v_x_166_){
_start:
{
uint8_t v_res_167_; lean_object* v_r_168_; 
v_res_167_ = lp_Quadlean_instDecidableEqQuadray(v_x_165_, v_x_166_);
lean_dec_ref(v_x_166_);
lean_dec_ref(v_x_165_);
v_r_168_ = lean_box(v_res_167_);
return v_r_168_;
}
}
LEAN_EXPORT lean_object* lp_Quadlean_Quadray_add(lean_object* v_q_169_, lean_object* v_r_170_){
_start:
{
lean_object* v_a_171_; lean_object* v_b_172_; lean_object* v_c_173_; lean_object* v_d_174_; lean_object* v_a_175_; lean_object* v_b_176_; lean_object* v_c_177_; lean_object* v_d_178_; lean_object* v___x_180_; uint8_t v_isShared_181_; uint8_t v_isSharedCheck_189_; 
v_a_171_ = lean_ctor_get(v_q_169_, 0);
v_b_172_ = lean_ctor_get(v_q_169_, 1);
v_c_173_ = lean_ctor_get(v_q_169_, 2);
v_d_174_ = lean_ctor_get(v_q_169_, 3);
v_a_175_ = lean_ctor_get(v_r_170_, 0);
v_b_176_ = lean_ctor_get(v_r_170_, 1);
v_c_177_ = lean_ctor_get(v_r_170_, 2);
v_d_178_ = lean_ctor_get(v_r_170_, 3);
v_isSharedCheck_189_ = !lean_is_exclusive(v_r_170_);
if (v_isSharedCheck_189_ == 0)
{
v___x_180_ = v_r_170_;
v_isShared_181_ = v_isSharedCheck_189_;
goto v_resetjp_179_;
}
else
{
lean_inc(v_d_178_);
lean_inc(v_c_177_);
lean_inc(v_b_176_);
lean_inc(v_a_175_);
lean_dec(v_r_170_);
v___x_180_ = lean_box(0);
v_isShared_181_ = v_isSharedCheck_189_;
goto v_resetjp_179_;
}
v_resetjp_179_:
{
lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; lean_object* v___x_185_; lean_object* v___x_187_; 
v___x_182_ = lean_int_add(v_a_171_, v_a_175_);
lean_dec(v_a_175_);
v___x_183_ = lean_int_add(v_b_172_, v_b_176_);
lean_dec(v_b_176_);
v___x_184_ = lean_int_add(v_c_173_, v_c_177_);
lean_dec(v_c_177_);
v___x_185_ = lean_int_add(v_d_174_, v_d_178_);
lean_dec(v_d_178_);
if (v_isShared_181_ == 0)
{
lean_ctor_set(v___x_180_, 3, v___x_185_);
lean_ctor_set(v___x_180_, 2, v___x_184_);
lean_ctor_set(v___x_180_, 1, v___x_183_);
lean_ctor_set(v___x_180_, 0, v___x_182_);
v___x_187_ = v___x_180_;
goto v_reusejp_186_;
}
else
{
lean_object* v_reuseFailAlloc_188_; 
v_reuseFailAlloc_188_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_188_, 0, v___x_182_);
lean_ctor_set(v_reuseFailAlloc_188_, 1, v___x_183_);
lean_ctor_set(v_reuseFailAlloc_188_, 2, v___x_184_);
lean_ctor_set(v_reuseFailAlloc_188_, 3, v___x_185_);
v___x_187_ = v_reuseFailAlloc_188_;
goto v_reusejp_186_;
}
v_reusejp_186_:
{
return v___x_187_;
}
}
}
}
LEAN_EXPORT lean_object* lp_Quadlean_Quadray_add___boxed(lean_object* v_q_190_, lean_object* v_r_191_){
_start:
{
lean_object* v_res_192_; 
v_res_192_ = lp_Quadlean_Quadray_add(v_q_190_, v_r_191_);
lean_dec_ref(v_q_190_);
return v_res_192_;
}
}
LEAN_EXPORT lean_object* lp_Quadlean_Quadray_sub(lean_object* v_q_193_, lean_object* v_r_194_){
_start:
{
lean_object* v_a_195_; lean_object* v_b_196_; lean_object* v_c_197_; lean_object* v_d_198_; lean_object* v_a_199_; lean_object* v_b_200_; lean_object* v_c_201_; lean_object* v_d_202_; lean_object* v___x_204_; uint8_t v_isShared_205_; uint8_t v_isSharedCheck_213_; 
v_a_195_ = lean_ctor_get(v_q_193_, 0);
v_b_196_ = lean_ctor_get(v_q_193_, 1);
v_c_197_ = lean_ctor_get(v_q_193_, 2);
v_d_198_ = lean_ctor_get(v_q_193_, 3);
v_a_199_ = lean_ctor_get(v_r_194_, 0);
v_b_200_ = lean_ctor_get(v_r_194_, 1);
v_c_201_ = lean_ctor_get(v_r_194_, 2);
v_d_202_ = lean_ctor_get(v_r_194_, 3);
v_isSharedCheck_213_ = !lean_is_exclusive(v_r_194_);
if (v_isSharedCheck_213_ == 0)
{
v___x_204_ = v_r_194_;
v_isShared_205_ = v_isSharedCheck_213_;
goto v_resetjp_203_;
}
else
{
lean_inc(v_d_202_);
lean_inc(v_c_201_);
lean_inc(v_b_200_);
lean_inc(v_a_199_);
lean_dec(v_r_194_);
v___x_204_ = lean_box(0);
v_isShared_205_ = v_isSharedCheck_213_;
goto v_resetjp_203_;
}
v_resetjp_203_:
{
lean_object* v___x_206_; lean_object* v___x_207_; lean_object* v___x_208_; lean_object* v___x_209_; lean_object* v___x_211_; 
v___x_206_ = lean_int_sub(v_a_195_, v_a_199_);
lean_dec(v_a_199_);
v___x_207_ = lean_int_sub(v_b_196_, v_b_200_);
lean_dec(v_b_200_);
v___x_208_ = lean_int_sub(v_c_197_, v_c_201_);
lean_dec(v_c_201_);
v___x_209_ = lean_int_sub(v_d_198_, v_d_202_);
lean_dec(v_d_202_);
if (v_isShared_205_ == 0)
{
lean_ctor_set(v___x_204_, 3, v___x_209_);
lean_ctor_set(v___x_204_, 2, v___x_208_);
lean_ctor_set(v___x_204_, 1, v___x_207_);
lean_ctor_set(v___x_204_, 0, v___x_206_);
v___x_211_ = v___x_204_;
goto v_reusejp_210_;
}
else
{
lean_object* v_reuseFailAlloc_212_; 
v_reuseFailAlloc_212_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_212_, 0, v___x_206_);
lean_ctor_set(v_reuseFailAlloc_212_, 1, v___x_207_);
lean_ctor_set(v_reuseFailAlloc_212_, 2, v___x_208_);
lean_ctor_set(v_reuseFailAlloc_212_, 3, v___x_209_);
v___x_211_ = v_reuseFailAlloc_212_;
goto v_reusejp_210_;
}
v_reusejp_210_:
{
return v___x_211_;
}
}
}
}
LEAN_EXPORT lean_object* lp_Quadlean_Quadray_sub___boxed(lean_object* v_q_214_, lean_object* v_r_215_){
_start:
{
lean_object* v_res_216_; 
v_res_216_ = lp_Quadlean_Quadray_sub(v_q_214_, v_r_215_);
lean_dec_ref(v_q_214_);
return v_res_216_;
}
}
LEAN_EXPORT lean_object* lp_Quadlean_quadMin(lean_object* v_q_217_){
_start:
{
lean_object* v_a_218_; lean_object* v_b_219_; lean_object* v_c_220_; lean_object* v_d_221_; lean_object* v___y_223_; lean_object* v___y_226_; uint8_t v___x_228_; 
v_a_218_ = lean_ctor_get(v_q_217_, 0);
v_b_219_ = lean_ctor_get(v_q_217_, 1);
v_c_220_ = lean_ctor_get(v_q_217_, 2);
v_d_221_ = lean_ctor_get(v_q_217_, 3);
v___x_228_ = lean_int_dec_le(v_c_220_, v_d_221_);
if (v___x_228_ == 0)
{
v___y_226_ = v_d_221_;
goto v___jp_225_;
}
else
{
v___y_226_ = v_c_220_;
goto v___jp_225_;
}
v___jp_222_:
{
uint8_t v___x_224_; 
v___x_224_ = lean_int_dec_le(v_a_218_, v___y_223_);
if (v___x_224_ == 0)
{
lean_inc(v___y_223_);
return v___y_223_;
}
else
{
lean_inc(v_a_218_);
return v_a_218_;
}
}
v___jp_225_:
{
uint8_t v___x_227_; 
v___x_227_ = lean_int_dec_le(v_b_219_, v___y_226_);
if (v___x_227_ == 0)
{
v___y_223_ = v___y_226_;
goto v___jp_222_;
}
else
{
v___y_223_ = v_b_219_;
goto v___jp_222_;
}
}
}
}
LEAN_EXPORT lean_object* lp_Quadlean_quadMin___boxed(lean_object* v_q_229_){
_start:
{
lean_object* v_res_230_; 
v_res_230_ = lp_Quadlean_quadMin(v_q_229_);
lean_dec_ref(v_q_229_);
return v_res_230_;
}
}
LEAN_EXPORT lean_object* lp_Quadlean_Quadray_normalize(lean_object* v_q_231_){
_start:
{
lean_object* v_a_232_; lean_object* v_b_233_; lean_object* v_c_234_; lean_object* v_d_235_; lean_object* v___x_236_; lean_object* v___x_238_; uint8_t v_isShared_239_; uint8_t v_isSharedCheck_247_; 
v_a_232_ = lean_ctor_get(v_q_231_, 0);
lean_inc(v_a_232_);
v_b_233_ = lean_ctor_get(v_q_231_, 1);
lean_inc(v_b_233_);
v_c_234_ = lean_ctor_get(v_q_231_, 2);
lean_inc(v_c_234_);
v_d_235_ = lean_ctor_get(v_q_231_, 3);
lean_inc(v_d_235_);
v___x_236_ = lp_Quadlean_quadMin(v_q_231_);
v_isSharedCheck_247_ = !lean_is_exclusive(v_q_231_);
if (v_isSharedCheck_247_ == 0)
{
lean_object* v_unused_248_; lean_object* v_unused_249_; lean_object* v_unused_250_; lean_object* v_unused_251_; 
v_unused_248_ = lean_ctor_get(v_q_231_, 3);
lean_dec(v_unused_248_);
v_unused_249_ = lean_ctor_get(v_q_231_, 2);
lean_dec(v_unused_249_);
v_unused_250_ = lean_ctor_get(v_q_231_, 1);
lean_dec(v_unused_250_);
v_unused_251_ = lean_ctor_get(v_q_231_, 0);
lean_dec(v_unused_251_);
v___x_238_ = v_q_231_;
v_isShared_239_ = v_isSharedCheck_247_;
goto v_resetjp_237_;
}
else
{
lean_dec(v_q_231_);
v___x_238_ = lean_box(0);
v_isShared_239_ = v_isSharedCheck_247_;
goto v_resetjp_237_;
}
v_resetjp_237_:
{
lean_object* v___x_240_; lean_object* v___x_241_; lean_object* v___x_242_; lean_object* v___x_243_; lean_object* v___x_245_; 
v___x_240_ = lean_int_sub(v_a_232_, v___x_236_);
lean_dec(v_a_232_);
v___x_241_ = lean_int_sub(v_b_233_, v___x_236_);
lean_dec(v_b_233_);
v___x_242_ = lean_int_sub(v_c_234_, v___x_236_);
lean_dec(v_c_234_);
v___x_243_ = lean_int_sub(v_d_235_, v___x_236_);
lean_dec(v___x_236_);
lean_dec(v_d_235_);
if (v_isShared_239_ == 0)
{
lean_ctor_set(v___x_238_, 3, v___x_243_);
lean_ctor_set(v___x_238_, 2, v___x_242_);
lean_ctor_set(v___x_238_, 1, v___x_241_);
lean_ctor_set(v___x_238_, 0, v___x_240_);
v___x_245_ = v___x_238_;
goto v_reusejp_244_;
}
else
{
lean_object* v_reuseFailAlloc_246_; 
v_reuseFailAlloc_246_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_246_, 0, v___x_240_);
lean_ctor_set(v_reuseFailAlloc_246_, 1, v___x_241_);
lean_ctor_set(v_reuseFailAlloc_246_, 2, v___x_242_);
lean_ctor_set(v_reuseFailAlloc_246_, 3, v___x_243_);
v___x_245_ = v_reuseFailAlloc_246_;
goto v_reusejp_244_;
}
v_reusejp_244_:
{
return v___x_245_;
}
}
}
}
LEAN_EXPORT lean_object* lp_Quadlean_proj(lean_object* v_q_252_){
_start:
{
lean_object* v_a_253_; lean_object* v_b_254_; lean_object* v_c_255_; lean_object* v_d_256_; lean_object* v___x_257_; lean_object* v___x_258_; lean_object* v___x_259_; lean_object* v___x_260_; lean_object* v___x_261_; 
v_a_253_ = lean_ctor_get(v_q_252_, 0);
v_b_254_ = lean_ctor_get(v_q_252_, 1);
v_c_255_ = lean_ctor_get(v_q_252_, 2);
v_d_256_ = lean_ctor_get(v_q_252_, 3);
v___x_257_ = lean_int_sub(v_a_253_, v_d_256_);
v___x_258_ = lean_int_sub(v_b_254_, v_d_256_);
v___x_259_ = lean_int_sub(v_c_255_, v_d_256_);
v___x_260_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_260_, 0, v___x_258_);
lean_ctor_set(v___x_260_, 1, v___x_259_);
v___x_261_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_261_, 0, v___x_257_);
lean_ctor_set(v___x_261_, 1, v___x_260_);
return v___x_261_;
}
}
LEAN_EXPORT lean_object* lp_Quadlean_proj___boxed(lean_object* v_q_262_){
_start:
{
lean_object* v_res_263_; 
v_res_263_ = lp_Quadlean_proj(v_q_262_);
lean_dec_ref(v_q_262_);
return v_res_263_;
}
}
LEAN_EXPORT lean_object* lp_Quadlean_det3(lean_object* v_r1_264_, lean_object* v_r2_265_, lean_object* v_r3_266_){
_start:
{
lean_object* v_snd_267_; lean_object* v_snd_268_; lean_object* v_snd_269_; lean_object* v_fst_270_; lean_object* v_fst_271_; lean_object* v_fst_272_; lean_object* v_snd_273_; lean_object* v_fst_274_; lean_object* v_fst_275_; lean_object* v_snd_276_; lean_object* v_fst_277_; lean_object* v_snd_278_; lean_object* v___x_279_; lean_object* v___x_280_; lean_object* v___x_281_; lean_object* v___x_282_; lean_object* v___x_283_; lean_object* v___x_284_; lean_object* v___x_285_; lean_object* v___x_286_; lean_object* v___x_287_; lean_object* v___x_288_; lean_object* v___x_289_; lean_object* v___x_290_; lean_object* v___x_291_; lean_object* v___x_292_; 
v_snd_267_ = lean_ctor_get(v_r2_265_, 1);
v_snd_268_ = lean_ctor_get(v_r3_266_, 1);
v_snd_269_ = lean_ctor_get(v_r1_264_, 1);
v_fst_270_ = lean_ctor_get(v_r1_264_, 0);
v_fst_271_ = lean_ctor_get(v_r2_265_, 0);
v_fst_272_ = lean_ctor_get(v_snd_267_, 0);
v_snd_273_ = lean_ctor_get(v_snd_267_, 1);
v_fst_274_ = lean_ctor_get(v_r3_266_, 0);
v_fst_275_ = lean_ctor_get(v_snd_268_, 0);
v_snd_276_ = lean_ctor_get(v_snd_268_, 1);
v_fst_277_ = lean_ctor_get(v_snd_269_, 0);
v_snd_278_ = lean_ctor_get(v_snd_269_, 1);
v___x_279_ = lean_int_mul(v_fst_272_, v_snd_276_);
v___x_280_ = lean_int_mul(v_snd_273_, v_fst_275_);
v___x_281_ = lean_int_sub(v___x_279_, v___x_280_);
lean_dec(v___x_280_);
lean_dec(v___x_279_);
v___x_282_ = lean_int_mul(v_fst_270_, v___x_281_);
lean_dec(v___x_281_);
v___x_283_ = lean_int_mul(v_fst_271_, v_snd_276_);
v___x_284_ = lean_int_mul(v_snd_273_, v_fst_274_);
v___x_285_ = lean_int_sub(v___x_283_, v___x_284_);
lean_dec(v___x_284_);
lean_dec(v___x_283_);
v___x_286_ = lean_int_mul(v_fst_277_, v___x_285_);
lean_dec(v___x_285_);
v___x_287_ = lean_int_sub(v___x_282_, v___x_286_);
lean_dec(v___x_286_);
lean_dec(v___x_282_);
v___x_288_ = lean_int_mul(v_fst_271_, v_fst_275_);
v___x_289_ = lean_int_mul(v_fst_272_, v_fst_274_);
v___x_290_ = lean_int_sub(v___x_288_, v___x_289_);
lean_dec(v___x_289_);
lean_dec(v___x_288_);
v___x_291_ = lean_int_mul(v_snd_278_, v___x_290_);
lean_dec(v___x_290_);
v___x_292_ = lean_int_add(v___x_287_, v___x_291_);
lean_dec(v___x_291_);
lean_dec(v___x_287_);
return v___x_292_;
}
}
LEAN_EXPORT lean_object* lp_Quadlean_det3___boxed(lean_object* v_r1_293_, lean_object* v_r2_294_, lean_object* v_r3_295_){
_start:
{
lean_object* v_res_296_; 
v_res_296_ = lp_Quadlean_det3(v_r1_293_, v_r2_294_, v_r3_295_);
lean_dec_ref(v_r3_295_);
lean_dec_ref(v_r2_294_);
lean_dec_ref(v_r1_293_);
return v_res_296_;
}
}
LEAN_EXPORT lean_object* lp_Quadlean_tetraDet(lean_object* v_p0_297_, lean_object* v_p1_298_, lean_object* v_p2_299_, lean_object* v_p3_300_){
_start:
{
lean_object* v___x_301_; lean_object* v___x_302_; lean_object* v___x_303_; lean_object* v___x_304_; lean_object* v___x_305_; lean_object* v___x_306_; lean_object* v___x_307_; 
lean_inc_ref_n(v_p0_297_, 2);
v___x_301_ = lp_Quadlean_Quadray_sub(v_p1_298_, v_p0_297_);
v___x_302_ = lp_Quadlean_proj(v___x_301_);
lean_dec_ref(v___x_301_);
v___x_303_ = lp_Quadlean_Quadray_sub(v_p2_299_, v_p0_297_);
v___x_304_ = lp_Quadlean_proj(v___x_303_);
lean_dec_ref(v___x_303_);
v___x_305_ = lp_Quadlean_Quadray_sub(v_p3_300_, v_p0_297_);
v___x_306_ = lp_Quadlean_proj(v___x_305_);
lean_dec_ref(v___x_305_);
v___x_307_ = lp_Quadlean_det3(v___x_302_, v___x_304_, v___x_306_);
lean_dec_ref(v___x_306_);
lean_dec_ref(v___x_304_);
lean_dec_ref(v___x_302_);
return v___x_307_;
}
}
LEAN_EXPORT lean_object* lp_Quadlean_tetraDet___boxed(lean_object* v_p0_308_, lean_object* v_p1_309_, lean_object* v_p2_310_, lean_object* v_p3_311_){
_start:
{
lean_object* v_res_312_; 
v_res_312_ = lp_Quadlean_tetraDet(v_p0_308_, v_p1_309_, v_p2_310_, v_p3_311_);
lean_dec_ref(v_p3_311_);
lean_dec_ref(v_p2_310_);
lean_dec_ref(v_p1_309_);
return v_res_312_;
}
}
LEAN_EXPORT lean_object* lp_Quadlean_tetraVolume(lean_object* v_p0_313_, lean_object* v_p1_314_, lean_object* v_p2_315_, lean_object* v_p3_316_){
_start:
{
lean_object* v___x_317_; lean_object* v___x_318_; lean_object* v___x_319_; lean_object* v___x_320_; lean_object* v___x_321_; 
v___x_317_ = lp_Quadlean_tetraDet(v_p0_313_, v_p1_314_, v_p2_315_, v_p3_316_);
v___x_318_ = lean_nat_abs(v___x_317_);
lean_dec(v___x_317_);
v___x_319_ = lean_nat_to_int(v___x_318_);
v___x_320_ = lean_unsigned_to_nat(4u);
v___x_321_ = l_mkRat(v___x_319_, v___x_320_);
return v___x_321_;
}
}
LEAN_EXPORT lean_object* lp_Quadlean_tetraVolume___boxed(lean_object* v_p0_322_, lean_object* v_p1_323_, lean_object* v_p2_324_, lean_object* v_p3_325_){
_start:
{
lean_object* v_res_326_; 
v_res_326_ = lp_Quadlean_tetraVolume(v_p0_322_, v_p1_323_, v_p2_324_, v_p3_325_);
lean_dec_ref(v_p3_325_);
lean_dec_ref(v_p2_324_);
lean_dec_ref(v_p1_323_);
return v_res_326_;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_Quadlean_QuadMath_Quadray(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
