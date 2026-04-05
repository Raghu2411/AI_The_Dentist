// GENERATED CODE - DO NOT MODIFY BY HAND
// coverage:ignore-file
// ignore_for_file: type=lint
// ignore_for_file: unused_element, deprecated_member_use, deprecated_member_use_from_same_package, use_function_type_syntax_for_parameters, unnecessary_const, avoid_init_to_null, invalid_override_different_default_values_named, prefer_expression_function_bodies, annotate_overrides, invalid_annotation_target, unnecessary_question_mark

part of 'chatbot_cubit.dart';

// **************************************************************************
// FreezedGenerator
// **************************************************************************

// dart format off
T _$identity<T>(T value) => value;
/// @nodoc
mixin _$ChatbotState {

 List<Map<String, String>> get history; bool get isLoading; bool get isError; String? get error;
/// Create a copy of ChatbotState
/// with the given fields replaced by the non-null parameter values.
@JsonKey(includeFromJson: false, includeToJson: false)
@pragma('vm:prefer-inline')
$ChatbotStateCopyWith<ChatbotState> get copyWith => _$ChatbotStateCopyWithImpl<ChatbotState>(this as ChatbotState, _$identity);



@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is ChatbotState&&const DeepCollectionEquality().equals(other.history, history)&&(identical(other.isLoading, isLoading) || other.isLoading == isLoading)&&(identical(other.isError, isError) || other.isError == isError)&&(identical(other.error, error) || other.error == error));
}


@override
int get hashCode => Object.hash(runtimeType,const DeepCollectionEquality().hash(history),isLoading,isError,error);

@override
String toString() {
  return 'ChatbotState(history: $history, isLoading: $isLoading, isError: $isError, error: $error)';
}


}

/// @nodoc
abstract mixin class $ChatbotStateCopyWith<$Res>  {
  factory $ChatbotStateCopyWith(ChatbotState value, $Res Function(ChatbotState) _then) = _$ChatbotStateCopyWithImpl;
@useResult
$Res call({
 List<Map<String, String>> history, bool isLoading, bool isError, String? error
});




}
/// @nodoc
class _$ChatbotStateCopyWithImpl<$Res>
    implements $ChatbotStateCopyWith<$Res> {
  _$ChatbotStateCopyWithImpl(this._self, this._then);

  final ChatbotState _self;
  final $Res Function(ChatbotState) _then;

/// Create a copy of ChatbotState
/// with the given fields replaced by the non-null parameter values.
@pragma('vm:prefer-inline') @override $Res call({Object? history = null,Object? isLoading = null,Object? isError = null,Object? error = freezed,}) {
  return _then(_self.copyWith(
history: null == history ? _self.history : history // ignore: cast_nullable_to_non_nullable
as List<Map<String, String>>,isLoading: null == isLoading ? _self.isLoading : isLoading // ignore: cast_nullable_to_non_nullable
as bool,isError: null == isError ? _self.isError : isError // ignore: cast_nullable_to_non_nullable
as bool,error: freezed == error ? _self.error : error // ignore: cast_nullable_to_non_nullable
as String?,
  ));
}

}


/// Adds pattern-matching-related methods to [ChatbotState].
extension ChatbotStatePatterns on ChatbotState {
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

@optionalTypeArgs TResult maybeMap<TResult extends Object?>(TResult Function( _ChatbotState value)?  $default,{required TResult orElse(),}){
final _that = this;
switch (_that) {
case _ChatbotState() when $default != null:
return $default(_that);case _:
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

@optionalTypeArgs TResult map<TResult extends Object?>(TResult Function( _ChatbotState value)  $default,){
final _that = this;
switch (_that) {
case _ChatbotState():
return $default(_that);}
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

@optionalTypeArgs TResult? mapOrNull<TResult extends Object?>(TResult? Function( _ChatbotState value)?  $default,){
final _that = this;
switch (_that) {
case _ChatbotState() when $default != null:
return $default(_that);case _:
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

@optionalTypeArgs TResult maybeWhen<TResult extends Object?>(TResult Function( List<Map<String, String>> history,  bool isLoading,  bool isError,  String? error)?  $default,{required TResult orElse(),}) {final _that = this;
switch (_that) {
case _ChatbotState() when $default != null:
return $default(_that.history,_that.isLoading,_that.isError,_that.error);case _:
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

@optionalTypeArgs TResult when<TResult extends Object?>(TResult Function( List<Map<String, String>> history,  bool isLoading,  bool isError,  String? error)  $default,) {final _that = this;
switch (_that) {
case _ChatbotState():
return $default(_that.history,_that.isLoading,_that.isError,_that.error);}
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

@optionalTypeArgs TResult? whenOrNull<TResult extends Object?>(TResult? Function( List<Map<String, String>> history,  bool isLoading,  bool isError,  String? error)?  $default,) {final _that = this;
switch (_that) {
case _ChatbotState() when $default != null:
return $default(_that.history,_that.isLoading,_that.isError,_that.error);case _:
  return null;

}
}

}

/// @nodoc


class _ChatbotState implements ChatbotState {
  const _ChatbotState({final  List<Map<String, String>> history = const [], this.isLoading = false, this.isError = false, this.error}): _history = history;
  

 final  List<Map<String, String>> _history;
@override@JsonKey() List<Map<String, String>> get history {
  if (_history is EqualUnmodifiableListView) return _history;
  // ignore: implicit_dynamic_type
  return EqualUnmodifiableListView(_history);
}

@override@JsonKey() final  bool isLoading;
@override@JsonKey() final  bool isError;
@override final  String? error;

/// Create a copy of ChatbotState
/// with the given fields replaced by the non-null parameter values.
@override @JsonKey(includeFromJson: false, includeToJson: false)
@pragma('vm:prefer-inline')
_$ChatbotStateCopyWith<_ChatbotState> get copyWith => __$ChatbotStateCopyWithImpl<_ChatbotState>(this, _$identity);



@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is _ChatbotState&&const DeepCollectionEquality().equals(other._history, _history)&&(identical(other.isLoading, isLoading) || other.isLoading == isLoading)&&(identical(other.isError, isError) || other.isError == isError)&&(identical(other.error, error) || other.error == error));
}


@override
int get hashCode => Object.hash(runtimeType,const DeepCollectionEquality().hash(_history),isLoading,isError,error);

@override
String toString() {
  return 'ChatbotState(history: $history, isLoading: $isLoading, isError: $isError, error: $error)';
}


}

/// @nodoc
abstract mixin class _$ChatbotStateCopyWith<$Res> implements $ChatbotStateCopyWith<$Res> {
  factory _$ChatbotStateCopyWith(_ChatbotState value, $Res Function(_ChatbotState) _then) = __$ChatbotStateCopyWithImpl;
@override @useResult
$Res call({
 List<Map<String, String>> history, bool isLoading, bool isError, String? error
});




}
/// @nodoc
class __$ChatbotStateCopyWithImpl<$Res>
    implements _$ChatbotStateCopyWith<$Res> {
  __$ChatbotStateCopyWithImpl(this._self, this._then);

  final _ChatbotState _self;
  final $Res Function(_ChatbotState) _then;

/// Create a copy of ChatbotState
/// with the given fields replaced by the non-null parameter values.
@override @pragma('vm:prefer-inline') $Res call({Object? history = null,Object? isLoading = null,Object? isError = null,Object? error = freezed,}) {
  return _then(_ChatbotState(
history: null == history ? _self._history : history // ignore: cast_nullable_to_non_nullable
as List<Map<String, String>>,isLoading: null == isLoading ? _self.isLoading : isLoading // ignore: cast_nullable_to_non_nullable
as bool,isError: null == isError ? _self.isError : isError // ignore: cast_nullable_to_non_nullable
as bool,error: freezed == error ? _self.error : error // ignore: cast_nullable_to_non_nullable
as String?,
  ));
}


}

// dart format on
