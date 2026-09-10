// Lean compiler output
// Module: QuadMath.IVM
// Imports: public import Init public meta import Init public import QuadMath.Quadray
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
lean_object* l_List_reverse___redArg(lean_object*);
uint8_t lean_int_dec_le(lean_object*, lean_object*);
uint8_t lean_int_dec_eq(lean_object*, lean_object*);
lean_object* lean_int_add(lean_object*, lean_object*);
lean_object* lean_int_emod(lean_object*, lean_object*);
lean_object* lean_int_mul(lean_object*, lean_object*);
lean_object* lean_int_sub(lean_object*, lean_object*);
lean_object* lean_nat_abs(lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_List_range(lean_object*);
lean_object* l_List_foldl___at___00Array_appendList_spec__0___redArg(lean_object*, lean_object*);
static lean_once_cell_t lp_Quadlean_neighborMoves___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Quadlean_neighborMoves___closed__0;
static lean_once_cell_t lp_Quadlean_neighborMoves___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Quadlean_neighborMoves___closed__1;
static lean_once_cell_t lp_Quadlean_neighborMoves___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Quadlean_neighborMoves___closed__2;
static lean_once_cell_t lp_Quadlean_neighborMoves___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Quadlean_neighborMoves___closed__3;
static lean_once_cell_t lp_Quadlean_neighborMoves___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Quadlean_neighborMoves___closed__4;
static lean_once_cell_t lp_Quadlean_neighborMoves___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Quadlean_neighborMoves___closed__5;
static lean_once_cell_t lp_Quadlean_neighborMoves___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Quadlean_neighborMoves___closed__6;
static lean_once_cell_t lp_Quadlean_neighborMoves___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Quadlean_neighborMoves___closed__7;
static lean_once_cell_t lp_Quadlean_neighborMoves___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Quadlean_neighborMoves___closed__8;
static lean_once_cell_t lp_Quadlean_neighborMoves___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Quadlean_neighborMoves___closed__9;
static lean_once_cell_t lp_Quadlean_neighborMoves___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Quadlean_neighborMoves___closed__10;
static lean_once_cell_t lp_Quadlean_neighborMoves___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Quadlean_neighborMoves___closed__11;
static lean_once_cell_t lp_Quadlean_neighborMoves___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Quadlean_neighborMoves___closed__12;
static lean_once_cell_t lp_Quadlean_neighborMoves___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Quadlean_neighborMoves___closed__13;
static lean_once_cell_t lp_Quadlean_neighborMoves___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Quadlean_neighborMoves___closed__14;
static lean_once_cell_t lp_Quadlean_neighborMoves___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Quadlean_neighborMoves___closed__15;
static lean_once_cell_t lp_Quadlean_neighborMoves___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Quadlean_neighborMoves___closed__16;
static lean_once_cell_t lp_Quadlean_neighborMoves___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Quadlean_neighborMoves___closed__17;
static lean_once_cell_t lp_Quadlean_neighborMoves___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Quadlean_neighborMoves___closed__18;
static lean_once_cell_t lp_Quadlean_neighborMoves___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Quadlean_neighborMoves___closed__19;
static lean_once_cell_t lp_Quadlean_neighborMoves___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Quadlean_neighborMoves___closed__20;
static lean_once_cell_t lp_Quadlean_neighborMoves___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Quadlean_neighborMoves___closed__21;
static lean_once_cell_t lp_Quadlean_neighborMoves___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Quadlean_neighborMoves___closed__22;
static lean_once_cell_t lp_Quadlean_neighborMoves___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Quadlean_neighborMoves___closed__23;
static lean_once_cell_t lp_Quadlean_neighborMoves___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Quadlean_neighborMoves___closed__24;
static lean_once_cell_t lp_Quadlean_neighborMoves___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Quadlean_neighborMoves___closed__25;
static lean_once_cell_t lp_Quadlean_neighborMoves___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Quadlean_neighborMoves___closed__26;
LEAN_EXPORT lean_object* lp_Quadlean_neighborMoves;
LEAN_EXPORT lean_object* lp_Quadlean_ivmSum(lean_object*);
LEAN_EXPORT lean_object* lp_Quadlean_ivmSum___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_Quadlean_instDecidableIsNormalized(lean_object*);
LEAN_EXPORT lean_object* lp_Quadlean_instDecidableIsNormalized___boxed(lean_object*);
static lean_once_cell_t lp_Quadlean_instDecidableIsIVMSite___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Quadlean_instDecidableIsIVMSite___closed__0;
LEAN_EXPORT uint8_t lp_Quadlean_instDecidableIsIVMSite(lean_object*);
LEAN_EXPORT lean_object* lp_Quadlean_instDecidableIsIVMSite___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_Quadlean_shellNorm4(lean_object*);
LEAN_EXPORT lean_object* lp_Quadlean_shellNorm4___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_Quadlean_List_mapTR_loop___at___00List_mapTR_loop___at___00__private_QuadMath_IVM_0__intRange_spec__0_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Quadlean_List_mapTR_loop___at___00__private_QuadMath_IVM_0__intRange_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Quadlean___private_QuadMath_IVM_0__intRange(lean_object*);
LEAN_EXPORT lean_object* lp_Quadlean___private_QuadMath_IVM_0__intRange___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_Quadlean_List_mapTR_loop___at___00quadBox_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Quadlean___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00__private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00quadBox_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Quadlean___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00__private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00quadBox_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Quadlean___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00quadBox_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Quadlean___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00quadBox_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_Quadlean___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00__private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00quadBox_spec__2_spec__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_Quadlean___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00__private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00quadBox_spec__2_spec__3___closed__0 = (const lean_object*)&lp_Quadlean___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00__private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00quadBox_spec__2_spec__3___closed__0_value;
LEAN_EXPORT lean_object* lp_Quadlean___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00__private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00quadBox_spec__2_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Quadlean___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00__private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00quadBox_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Quadlean___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00quadBox_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Quadlean___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00quadBox_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Quadlean___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00quadBox_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Quadlean___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00quadBox_spec__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Quadlean_quadBox(lean_object*);
LEAN_EXPORT lean_object* lp_Quadlean_quadBox___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_Quadlean_List_filterTR_loop___at___00shellSites_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Quadlean_List_filterTR_loop___at___00shellSites_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Quadlean_shellSites(lean_object*);
LEAN_EXPORT lean_object* lp_Quadlean_shellSites___boxed(lean_object*);
static lean_object* _init_lp_Quadlean_neighborMoves___closed__0(void){
_start:
{
lean_object* v___x_1_; lean_object* v___x_2_; 
v___x_1_ = lean_unsigned_to_nat(0u);
v___x_2_ = lean_nat_to_int(v___x_1_);
return v___x_2_;
}
}
static lean_object* _init_lp_Quadlean_neighborMoves___closed__1(void){
_start:
{
lean_object* v___x_3_; lean_object* v___x_4_; 
v___x_3_ = lean_unsigned_to_nat(1u);
v___x_4_ = lean_nat_to_int(v___x_3_);
return v___x_4_;
}
}
static lean_object* _init_lp_Quadlean_neighborMoves___closed__2(void){
_start:
{
lean_object* v___x_5_; lean_object* v___x_6_; 
v___x_5_ = lean_unsigned_to_nat(2u);
v___x_6_ = lean_nat_to_int(v___x_5_);
return v___x_6_;
}
}
static lean_object* _init_lp_Quadlean_neighborMoves___closed__3(void){
_start:
{
lean_object* v___x_7_; lean_object* v___x_8_; lean_object* v___x_9_; lean_object* v___x_10_; 
v___x_7_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__2, &lp_Quadlean_neighborMoves___closed__2_once, _init_lp_Quadlean_neighborMoves___closed__2);
v___x_8_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__1, &lp_Quadlean_neighborMoves___closed__1_once, _init_lp_Quadlean_neighborMoves___closed__1);
v___x_9_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__0, &lp_Quadlean_neighborMoves___closed__0_once, _init_lp_Quadlean_neighborMoves___closed__0);
v___x_10_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_10_, 0, v___x_9_);
lean_ctor_set(v___x_10_, 1, v___x_8_);
lean_ctor_set(v___x_10_, 2, v___x_8_);
lean_ctor_set(v___x_10_, 3, v___x_7_);
return v___x_10_;
}
}
static lean_object* _init_lp_Quadlean_neighborMoves___closed__4(void){
_start:
{
lean_object* v___x_11_; lean_object* v___x_12_; lean_object* v___x_13_; lean_object* v___x_14_; 
v___x_11_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__2, &lp_Quadlean_neighborMoves___closed__2_once, _init_lp_Quadlean_neighborMoves___closed__2);
v___x_12_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__1, &lp_Quadlean_neighborMoves___closed__1_once, _init_lp_Quadlean_neighborMoves___closed__1);
v___x_13_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__0, &lp_Quadlean_neighborMoves___closed__0_once, _init_lp_Quadlean_neighborMoves___closed__0);
v___x_14_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_14_, 0, v___x_13_);
lean_ctor_set(v___x_14_, 1, v___x_12_);
lean_ctor_set(v___x_14_, 2, v___x_11_);
lean_ctor_set(v___x_14_, 3, v___x_12_);
return v___x_14_;
}
}
static lean_object* _init_lp_Quadlean_neighborMoves___closed__5(void){
_start:
{
lean_object* v___x_15_; lean_object* v___x_16_; lean_object* v___x_17_; lean_object* v___x_18_; 
v___x_15_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__1, &lp_Quadlean_neighborMoves___closed__1_once, _init_lp_Quadlean_neighborMoves___closed__1);
v___x_16_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__2, &lp_Quadlean_neighborMoves___closed__2_once, _init_lp_Quadlean_neighborMoves___closed__2);
v___x_17_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__0, &lp_Quadlean_neighborMoves___closed__0_once, _init_lp_Quadlean_neighborMoves___closed__0);
v___x_18_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_18_, 0, v___x_17_);
lean_ctor_set(v___x_18_, 1, v___x_16_);
lean_ctor_set(v___x_18_, 2, v___x_15_);
lean_ctor_set(v___x_18_, 3, v___x_15_);
return v___x_18_;
}
}
static lean_object* _init_lp_Quadlean_neighborMoves___closed__6(void){
_start:
{
lean_object* v___x_19_; lean_object* v___x_20_; lean_object* v___x_21_; lean_object* v___x_22_; 
v___x_19_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__2, &lp_Quadlean_neighborMoves___closed__2_once, _init_lp_Quadlean_neighborMoves___closed__2);
v___x_20_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__0, &lp_Quadlean_neighborMoves___closed__0_once, _init_lp_Quadlean_neighborMoves___closed__0);
v___x_21_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__1, &lp_Quadlean_neighborMoves___closed__1_once, _init_lp_Quadlean_neighborMoves___closed__1);
v___x_22_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_22_, 0, v___x_21_);
lean_ctor_set(v___x_22_, 1, v___x_20_);
lean_ctor_set(v___x_22_, 2, v___x_21_);
lean_ctor_set(v___x_22_, 3, v___x_19_);
return v___x_22_;
}
}
static lean_object* _init_lp_Quadlean_neighborMoves___closed__7(void){
_start:
{
lean_object* v___x_23_; lean_object* v___x_24_; lean_object* v___x_25_; lean_object* v___x_26_; 
v___x_23_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__2, &lp_Quadlean_neighborMoves___closed__2_once, _init_lp_Quadlean_neighborMoves___closed__2);
v___x_24_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__0, &lp_Quadlean_neighborMoves___closed__0_once, _init_lp_Quadlean_neighborMoves___closed__0);
v___x_25_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__1, &lp_Quadlean_neighborMoves___closed__1_once, _init_lp_Quadlean_neighborMoves___closed__1);
v___x_26_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_26_, 0, v___x_25_);
lean_ctor_set(v___x_26_, 1, v___x_24_);
lean_ctor_set(v___x_26_, 2, v___x_23_);
lean_ctor_set(v___x_26_, 3, v___x_25_);
return v___x_26_;
}
}
static lean_object* _init_lp_Quadlean_neighborMoves___closed__8(void){
_start:
{
lean_object* v___x_27_; lean_object* v___x_28_; lean_object* v___x_29_; lean_object* v___x_30_; 
v___x_27_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__2, &lp_Quadlean_neighborMoves___closed__2_once, _init_lp_Quadlean_neighborMoves___closed__2);
v___x_28_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__0, &lp_Quadlean_neighborMoves___closed__0_once, _init_lp_Quadlean_neighborMoves___closed__0);
v___x_29_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__1, &lp_Quadlean_neighborMoves___closed__1_once, _init_lp_Quadlean_neighborMoves___closed__1);
v___x_30_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_30_, 0, v___x_29_);
lean_ctor_set(v___x_30_, 1, v___x_29_);
lean_ctor_set(v___x_30_, 2, v___x_28_);
lean_ctor_set(v___x_30_, 3, v___x_27_);
return v___x_30_;
}
}
static lean_object* _init_lp_Quadlean_neighborMoves___closed__9(void){
_start:
{
lean_object* v___x_31_; lean_object* v___x_32_; lean_object* v___x_33_; lean_object* v___x_34_; 
v___x_31_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__0, &lp_Quadlean_neighborMoves___closed__0_once, _init_lp_Quadlean_neighborMoves___closed__0);
v___x_32_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__2, &lp_Quadlean_neighborMoves___closed__2_once, _init_lp_Quadlean_neighborMoves___closed__2);
v___x_33_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__1, &lp_Quadlean_neighborMoves___closed__1_once, _init_lp_Quadlean_neighborMoves___closed__1);
v___x_34_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_34_, 0, v___x_33_);
lean_ctor_set(v___x_34_, 1, v___x_33_);
lean_ctor_set(v___x_34_, 2, v___x_32_);
lean_ctor_set(v___x_34_, 3, v___x_31_);
return v___x_34_;
}
}
static lean_object* _init_lp_Quadlean_neighborMoves___closed__10(void){
_start:
{
lean_object* v___x_35_; lean_object* v___x_36_; lean_object* v___x_37_; lean_object* v___x_38_; 
v___x_35_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__0, &lp_Quadlean_neighborMoves___closed__0_once, _init_lp_Quadlean_neighborMoves___closed__0);
v___x_36_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__2, &lp_Quadlean_neighborMoves___closed__2_once, _init_lp_Quadlean_neighborMoves___closed__2);
v___x_37_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__1, &lp_Quadlean_neighborMoves___closed__1_once, _init_lp_Quadlean_neighborMoves___closed__1);
v___x_38_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_38_, 0, v___x_37_);
lean_ctor_set(v___x_38_, 1, v___x_36_);
lean_ctor_set(v___x_38_, 2, v___x_35_);
lean_ctor_set(v___x_38_, 3, v___x_37_);
return v___x_38_;
}
}
static lean_object* _init_lp_Quadlean_neighborMoves___closed__11(void){
_start:
{
lean_object* v___x_39_; lean_object* v___x_40_; lean_object* v___x_41_; lean_object* v___x_42_; 
v___x_39_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__0, &lp_Quadlean_neighborMoves___closed__0_once, _init_lp_Quadlean_neighborMoves___closed__0);
v___x_40_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__2, &lp_Quadlean_neighborMoves___closed__2_once, _init_lp_Quadlean_neighborMoves___closed__2);
v___x_41_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__1, &lp_Quadlean_neighborMoves___closed__1_once, _init_lp_Quadlean_neighborMoves___closed__1);
v___x_42_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_42_, 0, v___x_41_);
lean_ctor_set(v___x_42_, 1, v___x_40_);
lean_ctor_set(v___x_42_, 2, v___x_41_);
lean_ctor_set(v___x_42_, 3, v___x_39_);
return v___x_42_;
}
}
static lean_object* _init_lp_Quadlean_neighborMoves___closed__12(void){
_start:
{
lean_object* v___x_43_; lean_object* v___x_44_; lean_object* v___x_45_; lean_object* v___x_46_; 
v___x_43_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__1, &lp_Quadlean_neighborMoves___closed__1_once, _init_lp_Quadlean_neighborMoves___closed__1);
v___x_44_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__0, &lp_Quadlean_neighborMoves___closed__0_once, _init_lp_Quadlean_neighborMoves___closed__0);
v___x_45_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__2, &lp_Quadlean_neighborMoves___closed__2_once, _init_lp_Quadlean_neighborMoves___closed__2);
v___x_46_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_46_, 0, v___x_45_);
lean_ctor_set(v___x_46_, 1, v___x_44_);
lean_ctor_set(v___x_46_, 2, v___x_43_);
lean_ctor_set(v___x_46_, 3, v___x_43_);
return v___x_46_;
}
}
static lean_object* _init_lp_Quadlean_neighborMoves___closed__13(void){
_start:
{
lean_object* v___x_47_; lean_object* v___x_48_; lean_object* v___x_49_; lean_object* v___x_50_; 
v___x_47_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__0, &lp_Quadlean_neighborMoves___closed__0_once, _init_lp_Quadlean_neighborMoves___closed__0);
v___x_48_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__1, &lp_Quadlean_neighborMoves___closed__1_once, _init_lp_Quadlean_neighborMoves___closed__1);
v___x_49_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__2, &lp_Quadlean_neighborMoves___closed__2_once, _init_lp_Quadlean_neighborMoves___closed__2);
v___x_50_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_50_, 0, v___x_49_);
lean_ctor_set(v___x_50_, 1, v___x_48_);
lean_ctor_set(v___x_50_, 2, v___x_47_);
lean_ctor_set(v___x_50_, 3, v___x_48_);
return v___x_50_;
}
}
static lean_object* _init_lp_Quadlean_neighborMoves___closed__14(void){
_start:
{
lean_object* v___x_51_; lean_object* v___x_52_; lean_object* v___x_53_; lean_object* v___x_54_; 
v___x_51_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__0, &lp_Quadlean_neighborMoves___closed__0_once, _init_lp_Quadlean_neighborMoves___closed__0);
v___x_52_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__1, &lp_Quadlean_neighborMoves___closed__1_once, _init_lp_Quadlean_neighborMoves___closed__1);
v___x_53_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__2, &lp_Quadlean_neighborMoves___closed__2_once, _init_lp_Quadlean_neighborMoves___closed__2);
v___x_54_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_54_, 0, v___x_53_);
lean_ctor_set(v___x_54_, 1, v___x_52_);
lean_ctor_set(v___x_54_, 2, v___x_52_);
lean_ctor_set(v___x_54_, 3, v___x_51_);
return v___x_54_;
}
}
static lean_object* _init_lp_Quadlean_neighborMoves___closed__15(void){
_start:
{
lean_object* v___x_55_; lean_object* v___x_56_; lean_object* v___x_57_; 
v___x_55_ = lean_box(0);
v___x_56_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__14, &lp_Quadlean_neighborMoves___closed__14_once, _init_lp_Quadlean_neighborMoves___closed__14);
v___x_57_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_57_, 0, v___x_56_);
lean_ctor_set(v___x_57_, 1, v___x_55_);
return v___x_57_;
}
}
static lean_object* _init_lp_Quadlean_neighborMoves___closed__16(void){
_start:
{
lean_object* v___x_58_; lean_object* v___x_59_; lean_object* v___x_60_; 
v___x_58_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__15, &lp_Quadlean_neighborMoves___closed__15_once, _init_lp_Quadlean_neighborMoves___closed__15);
v___x_59_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__13, &lp_Quadlean_neighborMoves___closed__13_once, _init_lp_Quadlean_neighborMoves___closed__13);
v___x_60_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_60_, 0, v___x_59_);
lean_ctor_set(v___x_60_, 1, v___x_58_);
return v___x_60_;
}
}
static lean_object* _init_lp_Quadlean_neighborMoves___closed__17(void){
_start:
{
lean_object* v___x_61_; lean_object* v___x_62_; lean_object* v___x_63_; 
v___x_61_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__16, &lp_Quadlean_neighborMoves___closed__16_once, _init_lp_Quadlean_neighborMoves___closed__16);
v___x_62_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__12, &lp_Quadlean_neighborMoves___closed__12_once, _init_lp_Quadlean_neighborMoves___closed__12);
v___x_63_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_63_, 0, v___x_62_);
lean_ctor_set(v___x_63_, 1, v___x_61_);
return v___x_63_;
}
}
static lean_object* _init_lp_Quadlean_neighborMoves___closed__18(void){
_start:
{
lean_object* v___x_64_; lean_object* v___x_65_; lean_object* v___x_66_; 
v___x_64_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__17, &lp_Quadlean_neighborMoves___closed__17_once, _init_lp_Quadlean_neighborMoves___closed__17);
v___x_65_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__11, &lp_Quadlean_neighborMoves___closed__11_once, _init_lp_Quadlean_neighborMoves___closed__11);
v___x_66_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_66_, 0, v___x_65_);
lean_ctor_set(v___x_66_, 1, v___x_64_);
return v___x_66_;
}
}
static lean_object* _init_lp_Quadlean_neighborMoves___closed__19(void){
_start:
{
lean_object* v___x_67_; lean_object* v___x_68_; lean_object* v___x_69_; 
v___x_67_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__18, &lp_Quadlean_neighborMoves___closed__18_once, _init_lp_Quadlean_neighborMoves___closed__18);
v___x_68_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__10, &lp_Quadlean_neighborMoves___closed__10_once, _init_lp_Quadlean_neighborMoves___closed__10);
v___x_69_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_69_, 0, v___x_68_);
lean_ctor_set(v___x_69_, 1, v___x_67_);
return v___x_69_;
}
}
static lean_object* _init_lp_Quadlean_neighborMoves___closed__20(void){
_start:
{
lean_object* v___x_70_; lean_object* v___x_71_; lean_object* v___x_72_; 
v___x_70_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__19, &lp_Quadlean_neighborMoves___closed__19_once, _init_lp_Quadlean_neighborMoves___closed__19);
v___x_71_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__9, &lp_Quadlean_neighborMoves___closed__9_once, _init_lp_Quadlean_neighborMoves___closed__9);
v___x_72_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_72_, 0, v___x_71_);
lean_ctor_set(v___x_72_, 1, v___x_70_);
return v___x_72_;
}
}
static lean_object* _init_lp_Quadlean_neighborMoves___closed__21(void){
_start:
{
lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; 
v___x_73_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__20, &lp_Quadlean_neighborMoves___closed__20_once, _init_lp_Quadlean_neighborMoves___closed__20);
v___x_74_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__8, &lp_Quadlean_neighborMoves___closed__8_once, _init_lp_Quadlean_neighborMoves___closed__8);
v___x_75_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_75_, 0, v___x_74_);
lean_ctor_set(v___x_75_, 1, v___x_73_);
return v___x_75_;
}
}
static lean_object* _init_lp_Quadlean_neighborMoves___closed__22(void){
_start:
{
lean_object* v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; 
v___x_76_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__21, &lp_Quadlean_neighborMoves___closed__21_once, _init_lp_Quadlean_neighborMoves___closed__21);
v___x_77_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__7, &lp_Quadlean_neighborMoves___closed__7_once, _init_lp_Quadlean_neighborMoves___closed__7);
v___x_78_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_78_, 0, v___x_77_);
lean_ctor_set(v___x_78_, 1, v___x_76_);
return v___x_78_;
}
}
static lean_object* _init_lp_Quadlean_neighborMoves___closed__23(void){
_start:
{
lean_object* v___x_79_; lean_object* v___x_80_; lean_object* v___x_81_; 
v___x_79_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__22, &lp_Quadlean_neighborMoves___closed__22_once, _init_lp_Quadlean_neighborMoves___closed__22);
v___x_80_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__6, &lp_Quadlean_neighborMoves___closed__6_once, _init_lp_Quadlean_neighborMoves___closed__6);
v___x_81_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_81_, 0, v___x_80_);
lean_ctor_set(v___x_81_, 1, v___x_79_);
return v___x_81_;
}
}
static lean_object* _init_lp_Quadlean_neighborMoves___closed__24(void){
_start:
{
lean_object* v___x_82_; lean_object* v___x_83_; lean_object* v___x_84_; 
v___x_82_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__23, &lp_Quadlean_neighborMoves___closed__23_once, _init_lp_Quadlean_neighborMoves___closed__23);
v___x_83_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__5, &lp_Quadlean_neighborMoves___closed__5_once, _init_lp_Quadlean_neighborMoves___closed__5);
v___x_84_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_84_, 0, v___x_83_);
lean_ctor_set(v___x_84_, 1, v___x_82_);
return v___x_84_;
}
}
static lean_object* _init_lp_Quadlean_neighborMoves___closed__25(void){
_start:
{
lean_object* v___x_85_; lean_object* v___x_86_; lean_object* v___x_87_; 
v___x_85_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__24, &lp_Quadlean_neighborMoves___closed__24_once, _init_lp_Quadlean_neighborMoves___closed__24);
v___x_86_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__4, &lp_Quadlean_neighborMoves___closed__4_once, _init_lp_Quadlean_neighborMoves___closed__4);
v___x_87_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_87_, 0, v___x_86_);
lean_ctor_set(v___x_87_, 1, v___x_85_);
return v___x_87_;
}
}
static lean_object* _init_lp_Quadlean_neighborMoves___closed__26(void){
_start:
{
lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; 
v___x_88_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__25, &lp_Quadlean_neighborMoves___closed__25_once, _init_lp_Quadlean_neighborMoves___closed__25);
v___x_89_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__3, &lp_Quadlean_neighborMoves___closed__3_once, _init_lp_Quadlean_neighborMoves___closed__3);
v___x_90_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_90_, 0, v___x_89_);
lean_ctor_set(v___x_90_, 1, v___x_88_);
return v___x_90_;
}
}
static lean_object* _init_lp_Quadlean_neighborMoves(void){
_start:
{
lean_object* v___x_91_; 
v___x_91_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__26, &lp_Quadlean_neighborMoves___closed__26_once, _init_lp_Quadlean_neighborMoves___closed__26);
return v___x_91_;
}
}
LEAN_EXPORT lean_object* lp_Quadlean_ivmSum(lean_object* v_q_92_){
_start:
{
lean_object* v_a_93_; lean_object* v_b_94_; lean_object* v_c_95_; lean_object* v_d_96_; lean_object* v___x_97_; lean_object* v___x_98_; lean_object* v___x_99_; 
v_a_93_ = lean_ctor_get(v_q_92_, 0);
v_b_94_ = lean_ctor_get(v_q_92_, 1);
v_c_95_ = lean_ctor_get(v_q_92_, 2);
v_d_96_ = lean_ctor_get(v_q_92_, 3);
v___x_97_ = lean_int_add(v_a_93_, v_b_94_);
v___x_98_ = lean_int_add(v___x_97_, v_c_95_);
lean_dec(v___x_97_);
v___x_99_ = lean_int_add(v___x_98_, v_d_96_);
lean_dec(v___x_98_);
return v___x_99_;
}
}
LEAN_EXPORT lean_object* lp_Quadlean_ivmSum___boxed(lean_object* v_q_100_){
_start:
{
lean_object* v_res_101_; 
v_res_101_ = lp_Quadlean_ivmSum(v_q_100_);
lean_dec_ref(v_q_100_);
return v_res_101_;
}
}
LEAN_EXPORT uint8_t lp_Quadlean_instDecidableIsNormalized(lean_object* v_q_102_){
_start:
{
lean_object* v_a_103_; lean_object* v_b_104_; lean_object* v_c_105_; lean_object* v_d_106_; lean_object* v___x_107_; uint8_t v___x_108_; 
v_a_103_ = lean_ctor_get(v_q_102_, 0);
v_b_104_ = lean_ctor_get(v_q_102_, 1);
v_c_105_ = lean_ctor_get(v_q_102_, 2);
v_d_106_ = lean_ctor_get(v_q_102_, 3);
v___x_107_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__0, &lp_Quadlean_neighborMoves___closed__0_once, _init_lp_Quadlean_neighborMoves___closed__0);
v___x_108_ = lean_int_dec_le(v___x_107_, v_a_103_);
if (v___x_108_ == 0)
{
return v___x_108_;
}
else
{
uint8_t v___x_109_; 
v___x_109_ = lean_int_dec_le(v___x_107_, v_b_104_);
if (v___x_109_ == 0)
{
return v___x_109_;
}
else
{
uint8_t v___x_110_; 
v___x_110_ = lean_int_dec_le(v___x_107_, v_c_105_);
if (v___x_110_ == 0)
{
return v___x_110_;
}
else
{
uint8_t v___x_111_; 
v___x_111_ = lean_int_dec_le(v___x_107_, v_d_106_);
if (v___x_111_ == 0)
{
return v___x_111_;
}
else
{
uint8_t v___x_112_; 
v___x_112_ = lean_int_dec_eq(v_a_103_, v___x_107_);
if (v___x_112_ == 0)
{
uint8_t v___x_113_; 
v___x_113_ = lean_int_dec_eq(v_b_104_, v___x_107_);
if (v___x_113_ == 0)
{
uint8_t v___x_114_; 
v___x_114_ = lean_int_dec_eq(v_c_105_, v___x_107_);
if (v___x_114_ == 0)
{
uint8_t v___x_115_; 
v___x_115_ = lean_int_dec_eq(v_d_106_, v___x_107_);
return v___x_115_;
}
else
{
return v___x_114_;
}
}
else
{
return v___x_113_;
}
}
else
{
return v___x_112_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Quadlean_instDecidableIsNormalized___boxed(lean_object* v_q_116_){
_start:
{
uint8_t v_res_117_; lean_object* v_r_118_; 
v_res_117_ = lp_Quadlean_instDecidableIsNormalized(v_q_116_);
lean_dec_ref(v_q_116_);
v_r_118_ = lean_box(v_res_117_);
return v_r_118_;
}
}
static lean_object* _init_lp_Quadlean_instDecidableIsIVMSite___closed__0(void){
_start:
{
lean_object* v___x_119_; lean_object* v___x_120_; 
v___x_119_ = lean_unsigned_to_nat(4u);
v___x_120_ = lean_nat_to_int(v___x_119_);
return v___x_120_;
}
}
LEAN_EXPORT uint8_t lp_Quadlean_instDecidableIsIVMSite(lean_object* v_q_121_){
_start:
{
uint8_t v___x_122_; 
v___x_122_ = lp_Quadlean_instDecidableIsNormalized(v_q_121_);
if (v___x_122_ == 0)
{
return v___x_122_;
}
else
{
lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; uint8_t v___x_127_; 
v___x_123_ = lp_Quadlean_ivmSum(v_q_121_);
v___x_124_ = lean_obj_once(&lp_Quadlean_instDecidableIsIVMSite___closed__0, &lp_Quadlean_instDecidableIsIVMSite___closed__0_once, _init_lp_Quadlean_instDecidableIsIVMSite___closed__0);
v___x_125_ = lean_int_emod(v___x_123_, v___x_124_);
lean_dec(v___x_123_);
v___x_126_ = lean_obj_once(&lp_Quadlean_neighborMoves___closed__0, &lp_Quadlean_neighborMoves___closed__0_once, _init_lp_Quadlean_neighborMoves___closed__0);
v___x_127_ = lean_int_dec_eq(v___x_125_, v___x_126_);
lean_dec(v___x_125_);
return v___x_127_;
}
}
}
LEAN_EXPORT lean_object* lp_Quadlean_instDecidableIsIVMSite___boxed(lean_object* v_q_128_){
_start:
{
uint8_t v_res_129_; lean_object* v_r_130_; 
v_res_129_ = lp_Quadlean_instDecidableIsIVMSite(v_q_128_);
lean_dec_ref(v_q_128_);
v_r_130_ = lean_box(v_res_129_);
return v_r_130_;
}
}
LEAN_EXPORT lean_object* lp_Quadlean_shellNorm4(lean_object* v_q_131_){
_start:
{
lean_object* v_a_132_; lean_object* v_b_133_; lean_object* v_c_134_; lean_object* v_d_135_; lean_object* v___x_136_; lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; lean_object* v___x_147_; lean_object* v___x_148_; lean_object* v___x_149_; lean_object* v___x_150_; lean_object* v___x_151_; lean_object* v___x_152_; 
v_a_132_ = lean_ctor_get(v_q_131_, 0);
v_b_133_ = lean_ctor_get(v_q_131_, 1);
v_c_134_ = lean_ctor_get(v_q_131_, 2);
v_d_135_ = lean_ctor_get(v_q_131_, 3);
v___x_136_ = lean_obj_once(&lp_Quadlean_instDecidableIsIVMSite___closed__0, &lp_Quadlean_instDecidableIsIVMSite___closed__0_once, _init_lp_Quadlean_instDecidableIsIVMSite___closed__0);
v___x_137_ = lean_int_mul(v___x_136_, v_a_132_);
v___x_138_ = lp_Quadlean_ivmSum(v_q_131_);
v___x_139_ = lean_int_sub(v___x_137_, v___x_138_);
lean_dec(v___x_137_);
v___x_140_ = lean_nat_abs(v___x_139_);
lean_dec(v___x_139_);
v___x_141_ = lean_int_mul(v___x_136_, v_b_133_);
v___x_142_ = lean_int_sub(v___x_141_, v___x_138_);
lean_dec(v___x_141_);
v___x_143_ = lean_nat_abs(v___x_142_);
lean_dec(v___x_142_);
v___x_144_ = lean_nat_add(v___x_140_, v___x_143_);
lean_dec(v___x_143_);
lean_dec(v___x_140_);
v___x_145_ = lean_int_mul(v___x_136_, v_c_134_);
v___x_146_ = lean_int_sub(v___x_145_, v___x_138_);
lean_dec(v___x_145_);
v___x_147_ = lean_nat_abs(v___x_146_);
lean_dec(v___x_146_);
v___x_148_ = lean_nat_add(v___x_144_, v___x_147_);
lean_dec(v___x_147_);
lean_dec(v___x_144_);
v___x_149_ = lean_int_mul(v___x_136_, v_d_135_);
v___x_150_ = lean_int_sub(v___x_149_, v___x_138_);
lean_dec(v___x_138_);
lean_dec(v___x_149_);
v___x_151_ = lean_nat_abs(v___x_150_);
lean_dec(v___x_150_);
v___x_152_ = lean_nat_add(v___x_148_, v___x_151_);
lean_dec(v___x_151_);
lean_dec(v___x_148_);
return v___x_152_;
}
}
LEAN_EXPORT lean_object* lp_Quadlean_shellNorm4___boxed(lean_object* v_q_153_){
_start:
{
lean_object* v_res_154_; 
v_res_154_ = lp_Quadlean_shellNorm4(v_q_153_);
lean_dec_ref(v_q_153_);
return v_res_154_;
}
}
LEAN_EXPORT lean_object* lp_Quadlean_List_mapTR_loop___at___00List_mapTR_loop___at___00__private_QuadMath_IVM_0__intRange_spec__0_spec__0(lean_object* v_a_155_, lean_object* v_a_156_){
_start:
{
if (lean_obj_tag(v_a_155_) == 0)
{
lean_object* v___x_157_; 
v___x_157_ = l_List_reverse___redArg(v_a_156_);
return v___x_157_;
}
else
{
lean_object* v_head_158_; lean_object* v_tail_159_; lean_object* v___x_161_; uint8_t v_isShared_162_; uint8_t v_isSharedCheck_168_; 
v_head_158_ = lean_ctor_get(v_a_155_, 0);
v_tail_159_ = lean_ctor_get(v_a_155_, 1);
v_isSharedCheck_168_ = !lean_is_exclusive(v_a_155_);
if (v_isSharedCheck_168_ == 0)
{
v___x_161_ = v_a_155_;
v_isShared_162_ = v_isSharedCheck_168_;
goto v_resetjp_160_;
}
else
{
lean_inc(v_tail_159_);
lean_inc(v_head_158_);
lean_dec(v_a_155_);
v___x_161_ = lean_box(0);
v_isShared_162_ = v_isSharedCheck_168_;
goto v_resetjp_160_;
}
v_resetjp_160_:
{
lean_object* v___x_163_; lean_object* v___x_165_; 
v___x_163_ = lean_nat_to_int(v_head_158_);
if (v_isShared_162_ == 0)
{
lean_ctor_set(v___x_161_, 1, v_a_156_);
lean_ctor_set(v___x_161_, 0, v___x_163_);
v___x_165_ = v___x_161_;
goto v_reusejp_164_;
}
else
{
lean_object* v_reuseFailAlloc_167_; 
v_reuseFailAlloc_167_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_167_, 0, v___x_163_);
lean_ctor_set(v_reuseFailAlloc_167_, 1, v_a_156_);
v___x_165_ = v_reuseFailAlloc_167_;
goto v_reusejp_164_;
}
v_reusejp_164_:
{
v_a_155_ = v_tail_159_;
v_a_156_ = v___x_165_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Quadlean_List_mapTR_loop___at___00__private_QuadMath_IVM_0__intRange_spec__0(lean_object* v_a_169_, lean_object* v_a_170_){
_start:
{
if (lean_obj_tag(v_a_169_) == 0)
{
lean_object* v___x_171_; 
v___x_171_ = l_List_reverse___redArg(v_a_170_);
return v___x_171_;
}
else
{
lean_object* v_head_172_; lean_object* v_tail_173_; lean_object* v___x_175_; uint8_t v_isShared_176_; uint8_t v_isSharedCheck_182_; 
v_head_172_ = lean_ctor_get(v_a_169_, 0);
v_tail_173_ = lean_ctor_get(v_a_169_, 1);
v_isSharedCheck_182_ = !lean_is_exclusive(v_a_169_);
if (v_isSharedCheck_182_ == 0)
{
v___x_175_ = v_a_169_;
v_isShared_176_ = v_isSharedCheck_182_;
goto v_resetjp_174_;
}
else
{
lean_inc(v_tail_173_);
lean_inc(v_head_172_);
lean_dec(v_a_169_);
v___x_175_ = lean_box(0);
v_isShared_176_ = v_isSharedCheck_182_;
goto v_resetjp_174_;
}
v_resetjp_174_:
{
lean_object* v___x_177_; lean_object* v___x_179_; 
v___x_177_ = lean_nat_to_int(v_head_172_);
if (v_isShared_176_ == 0)
{
lean_ctor_set(v___x_175_, 1, v_a_170_);
lean_ctor_set(v___x_175_, 0, v___x_177_);
v___x_179_ = v___x_175_;
goto v_reusejp_178_;
}
else
{
lean_object* v_reuseFailAlloc_181_; 
v_reuseFailAlloc_181_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_181_, 0, v___x_177_);
lean_ctor_set(v_reuseFailAlloc_181_, 1, v_a_170_);
v___x_179_ = v_reuseFailAlloc_181_;
goto v_reusejp_178_;
}
v_reusejp_178_:
{
lean_object* v___x_180_; 
v___x_180_ = lp_Quadlean_List_mapTR_loop___at___00List_mapTR_loop___at___00__private_QuadMath_IVM_0__intRange_spec__0_spec__0(v_tail_173_, v___x_179_);
return v___x_180_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Quadlean___private_QuadMath_IVM_0__intRange(lean_object* v_n_183_){
_start:
{
lean_object* v___x_184_; lean_object* v___x_185_; lean_object* v___x_186_; lean_object* v___x_187_; lean_object* v___x_188_; 
v___x_184_ = lean_unsigned_to_nat(1u);
v___x_185_ = lean_nat_add(v_n_183_, v___x_184_);
v___x_186_ = l_List_range(v___x_185_);
v___x_187_ = lean_box(0);
v___x_188_ = lp_Quadlean_List_mapTR_loop___at___00__private_QuadMath_IVM_0__intRange_spec__0(v___x_186_, v___x_187_);
return v___x_188_;
}
}
LEAN_EXPORT lean_object* lp_Quadlean___private_QuadMath_IVM_0__intRange___boxed(lean_object* v_n_189_){
_start:
{
lean_object* v_res_190_; 
v_res_190_ = lp_Quadlean___private_QuadMath_IVM_0__intRange(v_n_189_);
lean_dec(v_n_189_);
return v_res_190_;
}
}
LEAN_EXPORT lean_object* lp_Quadlean_List_mapTR_loop___at___00quadBox_spec__0(lean_object* v_a_191_, lean_object* v_b_192_, lean_object* v_c_193_, lean_object* v_a_194_, lean_object* v_a_195_){
_start:
{
if (lean_obj_tag(v_a_194_) == 0)
{
lean_object* v___x_196_; 
lean_dec(v_c_193_);
lean_dec(v_b_192_);
lean_dec(v_a_191_);
v___x_196_ = l_List_reverse___redArg(v_a_195_);
return v___x_196_;
}
else
{
lean_object* v_head_197_; lean_object* v_tail_198_; lean_object* v___x_200_; uint8_t v_isShared_201_; uint8_t v_isSharedCheck_207_; 
v_head_197_ = lean_ctor_get(v_a_194_, 0);
v_tail_198_ = lean_ctor_get(v_a_194_, 1);
v_isSharedCheck_207_ = !lean_is_exclusive(v_a_194_);
if (v_isSharedCheck_207_ == 0)
{
v___x_200_ = v_a_194_;
v_isShared_201_ = v_isSharedCheck_207_;
goto v_resetjp_199_;
}
else
{
lean_inc(v_tail_198_);
lean_inc(v_head_197_);
lean_dec(v_a_194_);
v___x_200_ = lean_box(0);
v_isShared_201_ = v_isSharedCheck_207_;
goto v_resetjp_199_;
}
v_resetjp_199_:
{
lean_object* v___x_202_; lean_object* v___x_204_; 
lean_inc(v_c_193_);
lean_inc(v_b_192_);
lean_inc(v_a_191_);
v___x_202_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_202_, 0, v_a_191_);
lean_ctor_set(v___x_202_, 1, v_b_192_);
lean_ctor_set(v___x_202_, 2, v_c_193_);
lean_ctor_set(v___x_202_, 3, v_head_197_);
if (v_isShared_201_ == 0)
{
lean_ctor_set(v___x_200_, 1, v_a_195_);
lean_ctor_set(v___x_200_, 0, v___x_202_);
v___x_204_ = v___x_200_;
goto v_reusejp_203_;
}
else
{
lean_object* v_reuseFailAlloc_206_; 
v_reuseFailAlloc_206_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_206_, 0, v___x_202_);
lean_ctor_set(v_reuseFailAlloc_206_, 1, v_a_195_);
v___x_204_ = v_reuseFailAlloc_206_;
goto v_reusejp_203_;
}
v_reusejp_203_:
{
v_a_194_ = v_tail_198_;
v_a_195_ = v___x_204_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Quadlean___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00__private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00quadBox_spec__1_spec__1(lean_object* v_n_208_, lean_object* v_a_209_, lean_object* v_b_210_, lean_object* v_a_211_, lean_object* v_a_212_){
_start:
{
if (lean_obj_tag(v_a_211_) == 0)
{
lean_object* v___x_213_; 
lean_dec(v_b_210_);
lean_dec(v_a_209_);
v___x_213_ = lean_array_to_list(v_a_212_);
return v___x_213_;
}
else
{
lean_object* v_head_214_; lean_object* v_tail_215_; lean_object* v___x_216_; lean_object* v___x_217_; lean_object* v___x_218_; lean_object* v___x_219_; 
v_head_214_ = lean_ctor_get(v_a_211_, 0);
lean_inc(v_head_214_);
v_tail_215_ = lean_ctor_get(v_a_211_, 1);
lean_inc(v_tail_215_);
lean_dec_ref_known(v_a_211_, 2);
v___x_216_ = lp_Quadlean___private_QuadMath_IVM_0__intRange(v_n_208_);
v___x_217_ = lean_box(0);
lean_inc(v_b_210_);
lean_inc(v_a_209_);
v___x_218_ = lp_Quadlean_List_mapTR_loop___at___00quadBox_spec__0(v_a_209_, v_b_210_, v_head_214_, v___x_216_, v___x_217_);
v___x_219_ = l_List_foldl___at___00Array_appendList_spec__0___redArg(v_a_212_, v___x_218_);
v_a_211_ = v_tail_215_;
v_a_212_ = v___x_219_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_Quadlean___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00__private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00quadBox_spec__1_spec__1___boxed(lean_object* v_n_221_, lean_object* v_a_222_, lean_object* v_b_223_, lean_object* v_a_224_, lean_object* v_a_225_){
_start:
{
lean_object* v_res_226_; 
v_res_226_ = lp_Quadlean___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00__private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00quadBox_spec__1_spec__1(v_n_221_, v_a_222_, v_b_223_, v_a_224_, v_a_225_);
lean_dec(v_n_221_);
return v_res_226_;
}
}
LEAN_EXPORT lean_object* lp_Quadlean___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00quadBox_spec__1(lean_object* v_a_227_, lean_object* v_b_228_, lean_object* v_n_229_, lean_object* v_a_230_, lean_object* v_a_231_){
_start:
{
if (lean_obj_tag(v_a_230_) == 0)
{
lean_object* v___x_232_; 
lean_dec(v_b_228_);
lean_dec(v_a_227_);
v___x_232_ = lean_array_to_list(v_a_231_);
return v___x_232_;
}
else
{
lean_object* v_head_233_; lean_object* v_tail_234_; lean_object* v___x_235_; lean_object* v___x_236_; lean_object* v___x_237_; lean_object* v___x_238_; lean_object* v___x_239_; 
v_head_233_ = lean_ctor_get(v_a_230_, 0);
lean_inc(v_head_233_);
v_tail_234_ = lean_ctor_get(v_a_230_, 1);
lean_inc(v_tail_234_);
lean_dec_ref_known(v_a_230_, 2);
v___x_235_ = lp_Quadlean___private_QuadMath_IVM_0__intRange(v_n_229_);
v___x_236_ = lean_box(0);
lean_inc(v_b_228_);
lean_inc(v_a_227_);
v___x_237_ = lp_Quadlean_List_mapTR_loop___at___00quadBox_spec__0(v_a_227_, v_b_228_, v_head_233_, v___x_235_, v___x_236_);
v___x_238_ = l_List_foldl___at___00Array_appendList_spec__0___redArg(v_a_231_, v___x_237_);
v___x_239_ = lp_Quadlean___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00__private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00quadBox_spec__1_spec__1(v_n_229_, v_a_227_, v_b_228_, v_tail_234_, v___x_238_);
return v___x_239_;
}
}
}
LEAN_EXPORT lean_object* lp_Quadlean___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00quadBox_spec__1___boxed(lean_object* v_a_240_, lean_object* v_b_241_, lean_object* v_n_242_, lean_object* v_a_243_, lean_object* v_a_244_){
_start:
{
lean_object* v_res_245_; 
v_res_245_ = lp_Quadlean___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00quadBox_spec__1(v_a_240_, v_b_241_, v_n_242_, v_a_243_, v_a_244_);
lean_dec(v_n_242_);
return v_res_245_;
}
}
LEAN_EXPORT lean_object* lp_Quadlean___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00__private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00quadBox_spec__2_spec__3(lean_object* v_n_248_, lean_object* v_a_249_, lean_object* v_a_250_, lean_object* v_a_251_){
_start:
{
if (lean_obj_tag(v_a_250_) == 0)
{
lean_object* v___x_252_; 
lean_dec(v_a_249_);
v___x_252_ = lean_array_to_list(v_a_251_);
return v___x_252_;
}
else
{
lean_object* v_head_253_; lean_object* v_tail_254_; lean_object* v___x_255_; lean_object* v___x_256_; lean_object* v___x_257_; lean_object* v___x_258_; 
v_head_253_ = lean_ctor_get(v_a_250_, 0);
lean_inc(v_head_253_);
v_tail_254_ = lean_ctor_get(v_a_250_, 1);
lean_inc(v_tail_254_);
lean_dec_ref_known(v_a_250_, 2);
v___x_255_ = lp_Quadlean___private_QuadMath_IVM_0__intRange(v_n_248_);
v___x_256_ = ((lean_object*)(lp_Quadlean___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00__private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00quadBox_spec__2_spec__3___closed__0));
lean_inc(v_a_249_);
v___x_257_ = lp_Quadlean___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00quadBox_spec__1(v_a_249_, v_head_253_, v_n_248_, v___x_255_, v___x_256_);
v___x_258_ = l_List_foldl___at___00Array_appendList_spec__0___redArg(v_a_251_, v___x_257_);
v_a_250_ = v_tail_254_;
v_a_251_ = v___x_258_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_Quadlean___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00__private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00quadBox_spec__2_spec__3___boxed(lean_object* v_n_260_, lean_object* v_a_261_, lean_object* v_a_262_, lean_object* v_a_263_){
_start:
{
lean_object* v_res_264_; 
v_res_264_ = lp_Quadlean___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00__private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00quadBox_spec__2_spec__3(v_n_260_, v_a_261_, v_a_262_, v_a_263_);
lean_dec(v_n_260_);
return v_res_264_;
}
}
LEAN_EXPORT lean_object* lp_Quadlean___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00quadBox_spec__2(lean_object* v_a_265_, lean_object* v_n_266_, lean_object* v_a_267_, lean_object* v_a_268_){
_start:
{
if (lean_obj_tag(v_a_267_) == 0)
{
lean_object* v___x_269_; 
lean_dec(v_a_265_);
v___x_269_ = lean_array_to_list(v_a_268_);
return v___x_269_;
}
else
{
lean_object* v_head_270_; lean_object* v_tail_271_; lean_object* v___x_272_; lean_object* v___x_273_; lean_object* v___x_274_; lean_object* v___x_275_; lean_object* v___x_276_; 
v_head_270_ = lean_ctor_get(v_a_267_, 0);
lean_inc(v_head_270_);
v_tail_271_ = lean_ctor_get(v_a_267_, 1);
lean_inc(v_tail_271_);
lean_dec_ref_known(v_a_267_, 2);
v___x_272_ = lp_Quadlean___private_QuadMath_IVM_0__intRange(v_n_266_);
v___x_273_ = ((lean_object*)(lp_Quadlean___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00__private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00quadBox_spec__2_spec__3___closed__0));
lean_inc(v_a_265_);
v___x_274_ = lp_Quadlean___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00quadBox_spec__1(v_a_265_, v_head_270_, v_n_266_, v___x_272_, v___x_273_);
v___x_275_ = l_List_foldl___at___00Array_appendList_spec__0___redArg(v_a_268_, v___x_274_);
v___x_276_ = lp_Quadlean___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00__private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00quadBox_spec__2_spec__3(v_n_266_, v_a_265_, v_tail_271_, v___x_275_);
return v___x_276_;
}
}
}
LEAN_EXPORT lean_object* lp_Quadlean___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00quadBox_spec__2___boxed(lean_object* v_a_277_, lean_object* v_n_278_, lean_object* v_a_279_, lean_object* v_a_280_){
_start:
{
lean_object* v_res_281_; 
v_res_281_ = lp_Quadlean___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00quadBox_spec__2(v_a_277_, v_n_278_, v_a_279_, v_a_280_);
lean_dec(v_n_278_);
return v_res_281_;
}
}
LEAN_EXPORT lean_object* lp_Quadlean___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00quadBox_spec__3(lean_object* v_n_282_, lean_object* v_a_283_, lean_object* v_a_284_){
_start:
{
if (lean_obj_tag(v_a_283_) == 0)
{
lean_object* v___x_285_; 
v___x_285_ = lean_array_to_list(v_a_284_);
return v___x_285_;
}
else
{
lean_object* v_head_286_; lean_object* v_tail_287_; lean_object* v___x_288_; lean_object* v___x_289_; lean_object* v___x_290_; lean_object* v___x_291_; 
v_head_286_ = lean_ctor_get(v_a_283_, 0);
lean_inc(v_head_286_);
v_tail_287_ = lean_ctor_get(v_a_283_, 1);
lean_inc(v_tail_287_);
lean_dec_ref_known(v_a_283_, 2);
v___x_288_ = lp_Quadlean___private_QuadMath_IVM_0__intRange(v_n_282_);
v___x_289_ = ((lean_object*)(lp_Quadlean___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00__private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00quadBox_spec__2_spec__3___closed__0));
v___x_290_ = lp_Quadlean___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00quadBox_spec__2(v_head_286_, v_n_282_, v___x_288_, v___x_289_);
v___x_291_ = l_List_foldl___at___00Array_appendList_spec__0___redArg(v_a_284_, v___x_290_);
v_a_283_ = v_tail_287_;
v_a_284_ = v___x_291_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_Quadlean___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00quadBox_spec__3___boxed(lean_object* v_n_293_, lean_object* v_a_294_, lean_object* v_a_295_){
_start:
{
lean_object* v_res_296_; 
v_res_296_ = lp_Quadlean___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00quadBox_spec__3(v_n_293_, v_a_294_, v_a_295_);
lean_dec(v_n_293_);
return v_res_296_;
}
}
LEAN_EXPORT lean_object* lp_Quadlean_quadBox(lean_object* v_n_297_){
_start:
{
lean_object* v___x_298_; lean_object* v___x_299_; lean_object* v___x_300_; 
v___x_298_ = lp_Quadlean___private_QuadMath_IVM_0__intRange(v_n_297_);
v___x_299_ = ((lean_object*)(lp_Quadlean___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00__private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00quadBox_spec__2_spec__3___closed__0));
v___x_300_ = lp_Quadlean___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00quadBox_spec__3(v_n_297_, v___x_298_, v___x_299_);
return v___x_300_;
}
}
LEAN_EXPORT lean_object* lp_Quadlean_quadBox___boxed(lean_object* v_n_301_){
_start:
{
lean_object* v_res_302_; 
v_res_302_ = lp_Quadlean_quadBox(v_n_301_);
lean_dec(v_n_301_);
return v_res_302_;
}
}
LEAN_EXPORT lean_object* lp_Quadlean_List_filterTR_loop___at___00shellSites_spec__0(lean_object* v_k_303_, lean_object* v_a_304_, lean_object* v_a_305_){
_start:
{
if (lean_obj_tag(v_a_304_) == 0)
{
lean_object* v___x_306_; 
v___x_306_ = l_List_reverse___redArg(v_a_305_);
return v___x_306_;
}
else
{
lean_object* v_head_307_; lean_object* v_tail_308_; lean_object* v___x_310_; uint8_t v_isShared_311_; uint8_t v_isSharedCheck_323_; 
v_head_307_ = lean_ctor_get(v_a_304_, 0);
v_tail_308_ = lean_ctor_get(v_a_304_, 1);
v_isSharedCheck_323_ = !lean_is_exclusive(v_a_304_);
if (v_isSharedCheck_323_ == 0)
{
v___x_310_ = v_a_304_;
v_isShared_311_ = v_isSharedCheck_323_;
goto v_resetjp_309_;
}
else
{
lean_inc(v_tail_308_);
lean_inc(v_head_307_);
lean_dec(v_a_304_);
v___x_310_ = lean_box(0);
v_isShared_311_ = v_isSharedCheck_323_;
goto v_resetjp_309_;
}
v_resetjp_309_:
{
uint8_t v___x_312_; 
v___x_312_ = lp_Quadlean_instDecidableIsIVMSite(v_head_307_);
if (v___x_312_ == 0)
{
lean_del_object(v___x_310_);
lean_dec(v_head_307_);
v_a_304_ = v_tail_308_;
goto _start;
}
else
{
lean_object* v___x_314_; lean_object* v___x_315_; lean_object* v___x_316_; uint8_t v___x_317_; 
v___x_314_ = lp_Quadlean_shellNorm4(v_head_307_);
v___x_315_ = lean_unsigned_to_nat(8u);
v___x_316_ = lean_nat_mul(v___x_315_, v_k_303_);
v___x_317_ = lean_nat_dec_eq(v___x_314_, v___x_316_);
lean_dec(v___x_316_);
lean_dec(v___x_314_);
if (v___x_317_ == 0)
{
lean_del_object(v___x_310_);
lean_dec(v_head_307_);
v_a_304_ = v_tail_308_;
goto _start;
}
else
{
lean_object* v___x_320_; 
if (v_isShared_311_ == 0)
{
lean_ctor_set(v___x_310_, 1, v_a_305_);
v___x_320_ = v___x_310_;
goto v_reusejp_319_;
}
else
{
lean_object* v_reuseFailAlloc_322_; 
v_reuseFailAlloc_322_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_322_, 0, v_head_307_);
lean_ctor_set(v_reuseFailAlloc_322_, 1, v_a_305_);
v___x_320_ = v_reuseFailAlloc_322_;
goto v_reusejp_319_;
}
v_reusejp_319_:
{
v_a_304_ = v_tail_308_;
v_a_305_ = v___x_320_;
goto _start;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Quadlean_List_filterTR_loop___at___00shellSites_spec__0___boxed(lean_object* v_k_324_, lean_object* v_a_325_, lean_object* v_a_326_){
_start:
{
lean_object* v_res_327_; 
v_res_327_ = lp_Quadlean_List_filterTR_loop___at___00shellSites_spec__0(v_k_324_, v_a_325_, v_a_326_);
lean_dec(v_k_324_);
return v_res_327_;
}
}
LEAN_EXPORT lean_object* lp_Quadlean_shellSites(lean_object* v_k_328_){
_start:
{
lean_object* v___x_329_; lean_object* v___x_330_; lean_object* v___x_331_; lean_object* v___x_332_; lean_object* v___x_333_; 
v___x_329_ = lean_unsigned_to_nat(2u);
v___x_330_ = lean_nat_mul(v___x_329_, v_k_328_);
v___x_331_ = lp_Quadlean_quadBox(v___x_330_);
lean_dec(v___x_330_);
v___x_332_ = lean_box(0);
v___x_333_ = lp_Quadlean_List_filterTR_loop___at___00shellSites_spec__0(v_k_328_, v___x_331_, v___x_332_);
return v___x_333_;
}
}
LEAN_EXPORT lean_object* lp_Quadlean_shellSites___boxed(lean_object* v_k_334_){
_start:
{
lean_object* v_res_335_; 
v_res_335_ = lp_Quadlean_shellSites(v_k_334_);
lean_dec(v_k_334_);
return v_res_335_;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Quadlean_QuadMath_Quadray(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_Quadlean_QuadMath_IVM(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Quadlean_QuadMath_Quadray(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_Quadlean_neighborMoves = _init_lp_Quadlean_neighborMoves();
lean_mark_persistent(lp_Quadlean_neighborMoves);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
