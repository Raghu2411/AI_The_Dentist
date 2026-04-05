// GENERATED CODE - DO NOT MODIFY BY HAND
// coverage:ignore-file
// ignore_for_file: type=lint
// ignore_for_file: unused_element, deprecated_member_use, deprecated_member_use_from_same_package, use_function_type_syntax_for_parameters, unnecessary_const, avoid_init_to_null, invalid_override_different_default_values_named, prefer_expression_function_bodies, annotate_overrides, invalid_annotation_target, unnecessary_question_mark

part of 'share_file_cubit.dart';

// **************************************************************************
// FreezedGenerator
// **************************************************************************

// dart format off
T _$identity<T>(T value) => value;
/// @nodoc
mixin _$ShareFileState {





@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is ShareFileState);
}


@override
int get hashCode => runtimeType.hashCode;

@override
String toString() {
  return 'ShareFileState()';
}


}

/// @nodoc
class $ShareFileStateCopyWith<$Res>  {
$ShareFileStateCopyWith(ShareFileState _, $Res Function(ShareFileState) __);
}


/// Adds pattern-matching-related methods to [ShareFileState].
extension ShareFileStatePatterns on ShareFileState {
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

@optionalTypeArgs TResult maybeMap<TResult extends Object?>({TResult Function( _Initial value)?  initial,TResult Function( _Loading value)?  loading,TResult Function( _Error value)?  error,TResult Function( _DownLoading value)?  downLoading,TResult Function( _DownLoaded value)?  downLoaded,TResult Function( _DownLoadingError value)?  downLoadingError,TResult Function( _FileShared value)?  fileShared,required TResult orElse(),}){
final _that = this;
switch (_that) {
case _Initial() when initial != null:
return initial(_that);case _Loading() when loading != null:
return loading(_that);case _Error() when error != null:
return error(_that);case _DownLoading() when downLoading != null:
return downLoading(_that);case _DownLoaded() when downLoaded != null:
return downLoaded(_that);case _DownLoadingError() when downLoadingError != null:
return downLoadingError(_that);case _FileShared() when fileShared != null:
return fileShared(_that);case _:
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

@optionalTypeArgs TResult map<TResult extends Object?>({required TResult Function( _Initial value)  initial,required TResult Function( _Loading value)  loading,required TResult Function( _Error value)  error,required TResult Function( _DownLoading value)  downLoading,required TResult Function( _DownLoaded value)  downLoaded,required TResult Function( _DownLoadingError value)  downLoadingError,required TResult Function( _FileShared value)  fileShared,}){
final _that = this;
switch (_that) {
case _Initial():
return initial(_that);case _Loading():
return loading(_that);case _Error():
return error(_that);case _DownLoading():
return downLoading(_that);case _DownLoaded():
return downLoaded(_that);case _DownLoadingError():
return downLoadingError(_that);case _FileShared():
return fileShared(_that);}
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

@optionalTypeArgs TResult? mapOrNull<TResult extends Object?>({TResult? Function( _Initial value)?  initial,TResult? Function( _Loading value)?  loading,TResult? Function( _Error value)?  error,TResult? Function( _DownLoading value)?  downLoading,TResult? Function( _DownLoaded value)?  downLoaded,TResult? Function( _DownLoadingError value)?  downLoadingError,TResult? Function( _FileShared value)?  fileShared,}){
final _that = this;
switch (_that) {
case _Initial() when initial != null:
return initial(_that);case _Loading() when loading != null:
return loading(_that);case _Error() when error != null:
return error(_that);case _DownLoading() when downLoading != null:
return downLoading(_that);case _DownLoaded() when downLoaded != null:
return downLoaded(_that);case _DownLoadingError() when downLoadingError != null:
return downLoadingError(_that);case _FileShared() when fileShared != null:
return fileShared(_that);case _:
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

@optionalTypeArgs TResult maybeWhen<TResult extends Object?>({TResult Function()?  initial,TResult Function()?  loading,TResult Function( String message)?  error,TResult Function()?  downLoading,TResult Function( String filePath)?  downLoaded,TResult Function( String message)?  downLoadingError,TResult Function()?  fileShared,required TResult orElse(),}) {final _that = this;
switch (_that) {
case _Initial() when initial != null:
return initial();case _Loading() when loading != null:
return loading();case _Error() when error != null:
return error(_that.message);case _DownLoading() when downLoading != null:
return downLoading();case _DownLoaded() when downLoaded != null:
return downLoaded(_that.filePath);case _DownLoadingError() when downLoadingError != null:
return downLoadingError(_that.message);case _FileShared() when fileShared != null:
return fileShared();case _:
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

@optionalTypeArgs TResult when<TResult extends Object?>({required TResult Function()  initial,required TResult Function()  loading,required TResult Function( String message)  error,required TResult Function()  downLoading,required TResult Function( String filePath)  downLoaded,required TResult Function( String message)  downLoadingError,required TResult Function()  fileShared,}) {final _that = this;
switch (_that) {
case _Initial():
return initial();case _Loading():
return loading();case _Error():
return error(_that.message);case _DownLoading():
return downLoading();case _DownLoaded():
return downLoaded(_that.filePath);case _DownLoadingError():
return downLoadingError(_that.message);case _FileShared():
return fileShared();}
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

@optionalTypeArgs TResult? whenOrNull<TResult extends Object?>({TResult? Function()?  initial,TResult? Function()?  loading,TResult? Function( String message)?  error,TResult? Function()?  downLoading,TResult? Function( String filePath)?  downLoaded,TResult? Function( String message)?  downLoadingError,TResult? Function()?  fileShared,}) {final _that = this;
switch (_that) {
case _Initial() when initial != null:
return initial();case _Loading() when loading != null:
return loading();case _Error() when error != null:
return error(_that.message);case _DownLoading() when downLoading != null:
return downLoading();case _DownLoaded() when downLoaded != null:
return downLoaded(_that.filePath);case _DownLoadingError() when downLoadingError != null:
return downLoadingError(_that.message);case _FileShared() when fileShared != null:
return fileShared();case _:
  return null;

}
}

}

/// @nodoc


class _Initial implements ShareFileState {
  const _Initial();
  






@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is _Initial);
}


@override
int get hashCode => runtimeType.hashCode;

@override
String toString() {
  return 'ShareFileState.initial()';
}


}




/// @nodoc


class _Loading implements ShareFileState {
  const _Loading();
  






@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is _Loading);
}


@override
int get hashCode => runtimeType.hashCode;

@override
String toString() {
  return 'ShareFileState.loading()';
}


}




/// @nodoc


class _Error implements ShareFileState {
  const _Error(this.message);
  

 final  String message;

/// Create a copy of ShareFileState
/// with the given fields replaced by the non-null parameter values.
@JsonKey(includeFromJson: false, includeToJson: false)
@pragma('vm:prefer-inline')
_$ErrorCopyWith<_Error> get copyWith => __$ErrorCopyWithImpl<_Error>(this, _$identity);



@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is _Error&&(identical(other.message, message) || other.message == message));
}


@override
int get hashCode => Object.hash(runtimeType,message);

@override
String toString() {
  return 'ShareFileState.error(message: $message)';
}


}

/// @nodoc
abstract mixin class _$ErrorCopyWith<$Res> implements $ShareFileStateCopyWith<$Res> {
  factory _$ErrorCopyWith(_Error value, $Res Function(_Error) _then) = __$ErrorCopyWithImpl;
@useResult
$Res call({
 String message
});




}
/// @nodoc
class __$ErrorCopyWithImpl<$Res>
    implements _$ErrorCopyWith<$Res> {
  __$ErrorCopyWithImpl(this._self, this._then);

  final _Error _self;
  final $Res Function(_Error) _then;

/// Create a copy of ShareFileState
/// with the given fields replaced by the non-null parameter values.
@pragma('vm:prefer-inline') $Res call({Object? message = null,}) {
  return _then(_Error(
null == message ? _self.message : message // ignore: cast_nullable_to_non_nullable
as String,
  ));
}


}

/// @nodoc


class _DownLoading implements ShareFileState {
  const _DownLoading();
  






@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is _DownLoading);
}


@override
int get hashCode => runtimeType.hashCode;

@override
String toString() {
  return 'ShareFileState.downLoading()';
}


}




/// @nodoc


class _DownLoaded implements ShareFileState {
  const _DownLoaded(this.filePath);
  

 final  String filePath;

/// Create a copy of ShareFileState
/// with the given fields replaced by the non-null parameter values.
@JsonKey(includeFromJson: false, includeToJson: false)
@pragma('vm:prefer-inline')
_$DownLoadedCopyWith<_DownLoaded> get copyWith => __$DownLoadedCopyWithImpl<_DownLoaded>(this, _$identity);



@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is _DownLoaded&&(identical(other.filePath, filePath) || other.filePath == filePath));
}


@override
int get hashCode => Object.hash(runtimeType,filePath);

@override
String toString() {
  return 'ShareFileState.downLoaded(filePath: $filePath)';
}


}

/// @nodoc
abstract mixin class _$DownLoadedCopyWith<$Res> implements $ShareFileStateCopyWith<$Res> {
  factory _$DownLoadedCopyWith(_DownLoaded value, $Res Function(_DownLoaded) _then) = __$DownLoadedCopyWithImpl;
@useResult
$Res call({
 String filePath
});




}
/// @nodoc
class __$DownLoadedCopyWithImpl<$Res>
    implements _$DownLoadedCopyWith<$Res> {
  __$DownLoadedCopyWithImpl(this._self, this._then);

  final _DownLoaded _self;
  final $Res Function(_DownLoaded) _then;

/// Create a copy of ShareFileState
/// with the given fields replaced by the non-null parameter values.
@pragma('vm:prefer-inline') $Res call({Object? filePath = null,}) {
  return _then(_DownLoaded(
null == filePath ? _self.filePath : filePath // ignore: cast_nullable_to_non_nullable
as String,
  ));
}


}

/// @nodoc


class _DownLoadingError implements ShareFileState {
  const _DownLoadingError(this.message);
  

 final  String message;

/// Create a copy of ShareFileState
/// with the given fields replaced by the non-null parameter values.
@JsonKey(includeFromJson: false, includeToJson: false)
@pragma('vm:prefer-inline')
_$DownLoadingErrorCopyWith<_DownLoadingError> get copyWith => __$DownLoadingErrorCopyWithImpl<_DownLoadingError>(this, _$identity);



@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is _DownLoadingError&&(identical(other.message, message) || other.message == message));
}


@override
int get hashCode => Object.hash(runtimeType,message);

@override
String toString() {
  return 'ShareFileState.downLoadingError(message: $message)';
}


}

/// @nodoc
abstract mixin class _$DownLoadingErrorCopyWith<$Res> implements $ShareFileStateCopyWith<$Res> {
  factory _$DownLoadingErrorCopyWith(_DownLoadingError value, $Res Function(_DownLoadingError) _then) = __$DownLoadingErrorCopyWithImpl;
@useResult
$Res call({
 String message
});




}
/// @nodoc
class __$DownLoadingErrorCopyWithImpl<$Res>
    implements _$DownLoadingErrorCopyWith<$Res> {
  __$DownLoadingErrorCopyWithImpl(this._self, this._then);

  final _DownLoadingError _self;
  final $Res Function(_DownLoadingError) _then;

/// Create a copy of ShareFileState
/// with the given fields replaced by the non-null parameter values.
@pragma('vm:prefer-inline') $Res call({Object? message = null,}) {
  return _then(_DownLoadingError(
null == message ? _self.message : message // ignore: cast_nullable_to_non_nullable
as String,
  ));
}


}

/// @nodoc


class _FileShared implements ShareFileState {
  const _FileShared();
  






@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is _FileShared);
}


@override
int get hashCode => runtimeType.hashCode;

@override
String toString() {
  return 'ShareFileState.fileShared()';
}


}




// dart format on
