// GENERATED CODE - DO NOT MODIFY BY HAND
// coverage:ignore-file
// ignore_for_file: type=lint
// ignore_for_file: unused_element, deprecated_member_use, deprecated_member_use_from_same_package, use_function_type_syntax_for_parameters, unnecessary_const, avoid_init_to_null, invalid_override_different_default_values_named, prefer_expression_function_bodies, annotate_overrides, invalid_annotation_target, unnecessary_question_mark

part of 'select_model_cubit.dart';

// **************************************************************************
// FreezedGenerator
// **************************************************************************

// dart format off
T _$identity<T>(T value) => value;
/// @nodoc
mixin _$SelectModelState {





@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is SelectModelState);
}


@override
int get hashCode => runtimeType.hashCode;

@override
String toString() {
  return 'SelectModelState()';
}


}

/// @nodoc
class $SelectModelStateCopyWith<$Res>  {
$SelectModelStateCopyWith(SelectModelState _, $Res Function(SelectModelState) __);
}


/// Adds pattern-matching-related methods to [SelectModelState].
extension SelectModelStatePatterns on SelectModelState {
/// A variant of `map` that fallback to returning `orElse`.
///
/// It is equivalent to doing:
/// ```dart
/// switch (sealedClass) {
///   case final Subclass value:
///     return ...;
///   case _:
///     return orElse();
/// }
/// ```

@optionalTypeArgs TResult maybeMap<TResult extends Object?>({TResult Function( _Initial value)?  initial,TResult Function( _SelectedModels value)?  selectedModels,required TResult orElse(),}){
final _that = this;
switch (_that) {
case _Initial() when initial != null:
return initial(_that);case _SelectedModels() when selectedModels != null:
return selectedModels(_that);case _:
  return orElse();

}
}
/// A `switch`-like method, using callbacks.
///
/// Callbacks receives the raw object, upcasted.
/// It is equivalent to doing:
/// ```dart
/// switch (sealedClass) {
///   case final Subclass value:
///     return ...;
///   case final Subclass2 value:
///     return ...;
/// }
/// ```

@optionalTypeArgs TResult map<TResult extends Object?>({required TResult Function( _Initial value)  initial,required TResult Function( _SelectedModels value)  selectedModels,}){
final _that = this;
switch (_that) {
case _Initial():
return initial(_that);case _SelectedModels():
return selectedModels(_that);case _:
  throw StateError('Unexpected subclass');

}
}
/// A variant of `map` that fallback to returning `null`.
///
/// It is equivalent to doing:
/// ```dart
/// switch (sealedClass) {
///   case final Subclass value:
///     return ...;
///   case _:
///     return null;
/// }
/// ```

@optionalTypeArgs TResult? mapOrNull<TResult extends Object?>({TResult? Function( _Initial value)?  initial,TResult? Function( _SelectedModels value)?  selectedModels,}){
final _that = this;
switch (_that) {
case _Initial() when initial != null:
return initial(_that);case _SelectedModels() when selectedModels != null:
return selectedModels(_that);case _:
  return null;

}
}
/// A variant of `when` that fallback to an `orElse` callback.
///
/// It is equivalent to doing:
/// ```dart
/// switch (sealedClass) {
///   case Subclass(:final field):
///     return ...;
///   case _:
///     return orElse();
/// }
/// ```

@optionalTypeArgs TResult maybeWhen<TResult extends Object?>({TResult Function()?  initial,TResult Function( List<String> selectedModels)?  selectedModels,required TResult orElse(),}) {final _that = this;
switch (_that) {
case _Initial() when initial != null:
return initial();case _SelectedModels() when selectedModels != null:
return selectedModels(_that.selectedModels);case _:
  return orElse();

}
}
/// A `switch`-like method, using callbacks.
///
/// As opposed to `map`, this offers destructuring.
/// It is equivalent to doing:
/// ```dart
/// switch (sealedClass) {
///   case Subclass(:final field):
///     return ...;
///   case Subclass2(:final field2):
///     return ...;
/// }
/// ```

@optionalTypeArgs TResult when<TResult extends Object?>({required TResult Function()  initial,required TResult Function( List<String> selectedModels)  selectedModels,}) {final _that = this;
switch (_that) {
case _Initial():
return initial();case _SelectedModels():
return selectedModels(_that.selectedModels);case _:
  throw StateError('Unexpected subclass');

}
}
/// A variant of `when` that fallback to returning `null`
///
/// It is equivalent to doing:
/// ```dart
/// switch (sealedClass) {
///   case Subclass(:final field):
///     return ...;
///   case _:
///     return null;
/// }
/// ```

@optionalTypeArgs TResult? whenOrNull<TResult extends Object?>({TResult? Function()?  initial,TResult? Function( List<String> selectedModels)?  selectedModels,}) {final _that = this;
switch (_that) {
case _Initial() when initial != null:
return initial();case _SelectedModels() when selectedModels != null:
return selectedModels(_that.selectedModels);case _:
  return null;

}
}

}

/// @nodoc


class _Initial implements SelectModelState {
  const _Initial();
  






@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is _Initial);
}


@override
int get hashCode => runtimeType.hashCode;

@override
String toString() {
  return 'SelectModelState.initial()';
}


}




/// @nodoc


class _SelectedModels implements SelectModelState {
  const _SelectedModels({final  List<String> selectedModels = const [Strings.defaultModel]}): _selectedModels = selectedModels;
  

 final  List<String> _selectedModels;
@JsonKey() List<String> get selectedModels {
  if (_selectedModels is EqualUnmodifiableListView) return _selectedModels;
  // ignore: implicit_dynamic_type
  return EqualUnmodifiableListView(_selectedModels);
}


/// Create a copy of SelectModelState
/// with the given fields replaced by the non-null parameter values.
@JsonKey(includeFromJson: false, includeToJson: false)
@pragma('vm:prefer-inline')
_$SelectedModelsCopyWith<_SelectedModels> get copyWith => __$SelectedModelsCopyWithImpl<_SelectedModels>(this, _$identity);



@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is _SelectedModels&&const DeepCollectionEquality().equals(other._selectedModels, _selectedModels));
}


@override
int get hashCode => Object.hash(runtimeType,const DeepCollectionEquality().hash(_selectedModels));

@override
String toString() {
  return 'SelectModelState.selectedModels(selectedModels: $selectedModels)';
}


}

/// @nodoc
abstract mixin class _$SelectedModelsCopyWith<$Res> implements $SelectModelStateCopyWith<$Res> {
  factory _$SelectedModelsCopyWith(_SelectedModels value, $Res Function(_SelectedModels) _then) = __$SelectedModelsCopyWithImpl;
@useResult
$Res call({
 List<String> selectedModels
});




}
/// @nodoc
class __$SelectedModelsCopyWithImpl<$Res>
    implements _$SelectedModelsCopyWith<$Res> {
  __$SelectedModelsCopyWithImpl(this._self, this._then);

  final _SelectedModels _self;
  final $Res Function(_SelectedModels) _then;

/// Create a copy of SelectModelState
/// with the given fields replaced by the non-null parameter values.
@pragma('vm:prefer-inline') $Res call({Object? selectedModels = null,}) {
  return _then(_SelectedModels(
selectedModels: null == selectedModels ? _self._selectedModels : selectedModels // ignore: cast_nullable_to_non_nullable
as List<String>,
  ));
}


}

// dart format on
