// GENERATED CODE - DO NOT MODIFY BY HAND
// coverage:ignore-file
// ignore_for_file: type=lint
// ignore_for_file: unused_element, deprecated_member_use, deprecated_member_use_from_same_package, use_function_type_syntax_for_parameters, unnecessary_const, avoid_init_to_null, invalid_override_different_default_values_named, prefer_expression_function_bodies, annotate_overrides, invalid_annotation_target, unnecessary_question_mark

part of 'select_image_cubit.dart';

// **************************************************************************
// FreezedGenerator
// **************************************************************************

// dart format off
T _$identity<T>(T value) => value;
/// @nodoc
mixin _$SelectImageState {





@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is SelectImageState);
}


@override
int get hashCode => runtimeType.hashCode;

@override
String toString() {
  return 'SelectImageState()';
}


}

/// @nodoc
class $SelectImageStateCopyWith<$Res>  {
$SelectImageStateCopyWith(SelectImageState _, $Res Function(SelectImageState) __);
}


/// Adds pattern-matching-related methods to [SelectImageState].
extension SelectImageStatePatterns on SelectImageState {
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

@optionalTypeArgs TResult maybeMap<TResult extends Object?>({TResult Function( _Initial value)?  initial,TResult Function( _FileTypeNotSupported value)?  fileTypeNotSupported,TResult Function( _AttachingImages value)?  attachingImages,TResult Function( _SelectedImages value)?  selectedImages,TResult Function( _Submitting value)?  submitting,TResult Function( _ImageAnalyzedSuccessfully value)?  imageAnalyzedSuccessfully,TResult Function( _Error value)?  error,TResult Function( _PreviewDocument value)?  previewImage,required TResult orElse(),}){
final _that = this;
switch (_that) {
case _Initial() when initial != null:
return initial(_that);case _FileTypeNotSupported() when fileTypeNotSupported != null:
return fileTypeNotSupported(_that);case _AttachingImages() when attachingImages != null:
return attachingImages(_that);case _SelectedImages() when selectedImages != null:
return selectedImages(_that);case _Submitting() when submitting != null:
return submitting(_that);case _ImageAnalyzedSuccessfully() when imageAnalyzedSuccessfully != null:
return imageAnalyzedSuccessfully(_that);case _Error() when error != null:
return error(_that);case _PreviewDocument() when previewImage != null:
return previewImage(_that);case _:
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

@optionalTypeArgs TResult map<TResult extends Object?>({required TResult Function( _Initial value)  initial,required TResult Function( _FileTypeNotSupported value)  fileTypeNotSupported,required TResult Function( _AttachingImages value)  attachingImages,required TResult Function( _SelectedImages value)  selectedImages,required TResult Function( _Submitting value)  submitting,required TResult Function( _ImageAnalyzedSuccessfully value)  imageAnalyzedSuccessfully,required TResult Function( _Error value)  error,required TResult Function( _PreviewDocument value)  previewImage,}){
final _that = this;
switch (_that) {
case _Initial():
return initial(_that);case _FileTypeNotSupported():
return fileTypeNotSupported(_that);case _AttachingImages():
return attachingImages(_that);case _SelectedImages():
return selectedImages(_that);case _Submitting():
return submitting(_that);case _ImageAnalyzedSuccessfully():
return imageAnalyzedSuccessfully(_that);case _Error():
return error(_that);case _PreviewDocument():
return previewImage(_that);case _:
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

@optionalTypeArgs TResult? mapOrNull<TResult extends Object?>({TResult? Function( _Initial value)?  initial,TResult? Function( _FileTypeNotSupported value)?  fileTypeNotSupported,TResult? Function( _AttachingImages value)?  attachingImages,TResult? Function( _SelectedImages value)?  selectedImages,TResult? Function( _Submitting value)?  submitting,TResult? Function( _ImageAnalyzedSuccessfully value)?  imageAnalyzedSuccessfully,TResult? Function( _Error value)?  error,TResult? Function( _PreviewDocument value)?  previewImage,}){
final _that = this;
switch (_that) {
case _Initial() when initial != null:
return initial(_that);case _FileTypeNotSupported() when fileTypeNotSupported != null:
return fileTypeNotSupported(_that);case _AttachingImages() when attachingImages != null:
return attachingImages(_that);case _SelectedImages() when selectedImages != null:
return selectedImages(_that);case _Submitting() when submitting != null:
return submitting(_that);case _ImageAnalyzedSuccessfully() when imageAnalyzedSuccessfully != null:
return imageAnalyzedSuccessfully(_that);case _Error() when error != null:
return error(_that);case _PreviewDocument() when previewImage != null:
return previewImage(_that);case _:
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

@optionalTypeArgs TResult maybeWhen<TResult extends Object?>({TResult Function()?  initial,TResult Function( String message)?  fileTypeNotSupported,TResult Function()?  attachingImages,TResult Function( List<String> selectedImagesPath)?  selectedImages,TResult Function()?  submitting,TResult Function( Analyses analyses,  ApiUrls apiUrls)?  imageAnalyzedSuccessfully,TResult Function( String message)?  error,TResult Function( String documentPath)?  previewImage,required TResult orElse(),}) {final _that = this;
switch (_that) {
case _Initial() when initial != null:
return initial();case _FileTypeNotSupported() when fileTypeNotSupported != null:
return fileTypeNotSupported(_that.message);case _AttachingImages() when attachingImages != null:
return attachingImages();case _SelectedImages() when selectedImages != null:
return selectedImages(_that.selectedImagesPath);case _Submitting() when submitting != null:
return submitting();case _ImageAnalyzedSuccessfully() when imageAnalyzedSuccessfully != null:
return imageAnalyzedSuccessfully(_that.analyses,_that.apiUrls);case _Error() when error != null:
return error(_that.message);case _PreviewDocument() when previewImage != null:
return previewImage(_that.documentPath);case _:
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

@optionalTypeArgs TResult when<TResult extends Object?>({required TResult Function()  initial,required TResult Function( String message)  fileTypeNotSupported,required TResult Function()  attachingImages,required TResult Function( List<String> selectedImagesPath)  selectedImages,required TResult Function()  submitting,required TResult Function( Analyses analyses,  ApiUrls apiUrls)  imageAnalyzedSuccessfully,required TResult Function( String message)  error,required TResult Function( String documentPath)  previewImage,}) {final _that = this;
switch (_that) {
case _Initial():
return initial();case _FileTypeNotSupported():
return fileTypeNotSupported(_that.message);case _AttachingImages():
return attachingImages();case _SelectedImages():
return selectedImages(_that.selectedImagesPath);case _Submitting():
return submitting();case _ImageAnalyzedSuccessfully():
return imageAnalyzedSuccessfully(_that.analyses,_that.apiUrls);case _Error():
return error(_that.message);case _PreviewDocument():
return previewImage(_that.documentPath);case _:
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

@optionalTypeArgs TResult? whenOrNull<TResult extends Object?>({TResult? Function()?  initial,TResult? Function( String message)?  fileTypeNotSupported,TResult? Function()?  attachingImages,TResult? Function( List<String> selectedImagesPath)?  selectedImages,TResult? Function()?  submitting,TResult? Function( Analyses analyses,  ApiUrls apiUrls)?  imageAnalyzedSuccessfully,TResult? Function( String message)?  error,TResult? Function( String documentPath)?  previewImage,}) {final _that = this;
switch (_that) {
case _Initial() when initial != null:
return initial();case _FileTypeNotSupported() when fileTypeNotSupported != null:
return fileTypeNotSupported(_that.message);case _AttachingImages() when attachingImages != null:
return attachingImages();case _SelectedImages() when selectedImages != null:
return selectedImages(_that.selectedImagesPath);case _Submitting() when submitting != null:
return submitting();case _ImageAnalyzedSuccessfully() when imageAnalyzedSuccessfully != null:
return imageAnalyzedSuccessfully(_that.analyses,_that.apiUrls);case _Error() when error != null:
return error(_that.message);case _PreviewDocument() when previewImage != null:
return previewImage(_that.documentPath);case _:
  return null;

}
}

}

/// @nodoc


class _Initial implements SelectImageState {
  const _Initial();
  






@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is _Initial);
}


@override
int get hashCode => runtimeType.hashCode;

@override
String toString() {
  return 'SelectImageState.initial()';
}


}




/// @nodoc


class _FileTypeNotSupported implements SelectImageState {
  const _FileTypeNotSupported(this.message);
  

 final  String message;

/// Create a copy of SelectImageState
/// with the given fields replaced by the non-null parameter values.
@JsonKey(includeFromJson: false, includeToJson: false)
@pragma('vm:prefer-inline')
_$FileTypeNotSupportedCopyWith<_FileTypeNotSupported> get copyWith => __$FileTypeNotSupportedCopyWithImpl<_FileTypeNotSupported>(this, _$identity);



@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is _FileTypeNotSupported&&(identical(other.message, message) || other.message == message));
}


@override
int get hashCode => Object.hash(runtimeType,message);

@override
String toString() {
  return 'SelectImageState.fileTypeNotSupported(message: $message)';
}


}

/// @nodoc
abstract mixin class _$FileTypeNotSupportedCopyWith<$Res> implements $SelectImageStateCopyWith<$Res> {
  factory _$FileTypeNotSupportedCopyWith(_FileTypeNotSupported value, $Res Function(_FileTypeNotSupported) _then) = __$FileTypeNotSupportedCopyWithImpl;
@useResult
$Res call({
 String message
});




}
/// @nodoc
class __$FileTypeNotSupportedCopyWithImpl<$Res>
    implements _$FileTypeNotSupportedCopyWith<$Res> {
  __$FileTypeNotSupportedCopyWithImpl(this._self, this._then);

  final _FileTypeNotSupported _self;
  final $Res Function(_FileTypeNotSupported) _then;

/// Create a copy of SelectImageState
/// with the given fields replaced by the non-null parameter values.
@pragma('vm:prefer-inline') $Res call({Object? message = null,}) {
  return _then(_FileTypeNotSupported(
null == message ? _self.message : message // ignore: cast_nullable_to_non_nullable
as String,
  ));
}


}

/// @nodoc


class _AttachingImages implements SelectImageState {
  const _AttachingImages();
  






@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is _AttachingImages);
}


@override
int get hashCode => runtimeType.hashCode;

@override
String toString() {
  return 'SelectImageState.attachingImages()';
}


}




/// @nodoc


class _SelectedImages implements SelectImageState {
  const _SelectedImages(final  List<String> selectedImagesPath): _selectedImagesPath = selectedImagesPath;
  

 final  List<String> _selectedImagesPath;
 List<String> get selectedImagesPath {
  if (_selectedImagesPath is EqualUnmodifiableListView) return _selectedImagesPath;
  // ignore: implicit_dynamic_type
  return EqualUnmodifiableListView(_selectedImagesPath);
}


/// Create a copy of SelectImageState
/// with the given fields replaced by the non-null parameter values.
@JsonKey(includeFromJson: false, includeToJson: false)
@pragma('vm:prefer-inline')
_$SelectedImagesCopyWith<_SelectedImages> get copyWith => __$SelectedImagesCopyWithImpl<_SelectedImages>(this, _$identity);



@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is _SelectedImages&&const DeepCollectionEquality().equals(other._selectedImagesPath, _selectedImagesPath));
}


@override
int get hashCode => Object.hash(runtimeType,const DeepCollectionEquality().hash(_selectedImagesPath));

@override
String toString() {
  return 'SelectImageState.selectedImages(selectedImagesPath: $selectedImagesPath)';
}


}

/// @nodoc
abstract mixin class _$SelectedImagesCopyWith<$Res> implements $SelectImageStateCopyWith<$Res> {
  factory _$SelectedImagesCopyWith(_SelectedImages value, $Res Function(_SelectedImages) _then) = __$SelectedImagesCopyWithImpl;
@useResult
$Res call({
 List<String> selectedImagesPath
});




}
/// @nodoc
class __$SelectedImagesCopyWithImpl<$Res>
    implements _$SelectedImagesCopyWith<$Res> {
  __$SelectedImagesCopyWithImpl(this._self, this._then);

  final _SelectedImages _self;
  final $Res Function(_SelectedImages) _then;

/// Create a copy of SelectImageState
/// with the given fields replaced by the non-null parameter values.
@pragma('vm:prefer-inline') $Res call({Object? selectedImagesPath = null,}) {
  return _then(_SelectedImages(
null == selectedImagesPath ? _self._selectedImagesPath : selectedImagesPath // ignore: cast_nullable_to_non_nullable
as List<String>,
  ));
}


}

/// @nodoc


class _Submitting implements SelectImageState {
  const _Submitting();
  






@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is _Submitting);
}


@override
int get hashCode => runtimeType.hashCode;

@override
String toString() {
  return 'SelectImageState.submitting()';
}


}




/// @nodoc


class _ImageAnalyzedSuccessfully implements SelectImageState {
  const _ImageAnalyzedSuccessfully(this.analyses, this.apiUrls);
  

 final  Analyses analyses;
 final  ApiUrls apiUrls;

/// Create a copy of SelectImageState
/// with the given fields replaced by the non-null parameter values.
@JsonKey(includeFromJson: false, includeToJson: false)
@pragma('vm:prefer-inline')
_$ImageAnalyzedSuccessfullyCopyWith<_ImageAnalyzedSuccessfully> get copyWith => __$ImageAnalyzedSuccessfullyCopyWithImpl<_ImageAnalyzedSuccessfully>(this, _$identity);



@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is _ImageAnalyzedSuccessfully&&(identical(other.analyses, analyses) || other.analyses == analyses)&&(identical(other.apiUrls, apiUrls) || other.apiUrls == apiUrls));
}


@override
int get hashCode => Object.hash(runtimeType,analyses,apiUrls);

@override
String toString() {
  return 'SelectImageState.imageAnalyzedSuccessfully(analyses: $analyses, apiUrls: $apiUrls)';
}


}

/// @nodoc
abstract mixin class _$ImageAnalyzedSuccessfullyCopyWith<$Res> implements $SelectImageStateCopyWith<$Res> {
  factory _$ImageAnalyzedSuccessfullyCopyWith(_ImageAnalyzedSuccessfully value, $Res Function(_ImageAnalyzedSuccessfully) _then) = __$ImageAnalyzedSuccessfullyCopyWithImpl;
@useResult
$Res call({
 Analyses analyses, ApiUrls apiUrls
});




}
/// @nodoc
class __$ImageAnalyzedSuccessfullyCopyWithImpl<$Res>
    implements _$ImageAnalyzedSuccessfullyCopyWith<$Res> {
  __$ImageAnalyzedSuccessfullyCopyWithImpl(this._self, this._then);

  final _ImageAnalyzedSuccessfully _self;
  final $Res Function(_ImageAnalyzedSuccessfully) _then;

/// Create a copy of SelectImageState
/// with the given fields replaced by the non-null parameter values.
@pragma('vm:prefer-inline') $Res call({Object? analyses = null,Object? apiUrls = null,}) {
  return _then(_ImageAnalyzedSuccessfully(
null == analyses ? _self.analyses : analyses // ignore: cast_nullable_to_non_nullable
as Analyses,null == apiUrls ? _self.apiUrls : apiUrls // ignore: cast_nullable_to_non_nullable
as ApiUrls,
  ));
}


}

/// @nodoc


class _Error implements SelectImageState {
  const _Error(this.message);
  

 final  String message;

/// Create a copy of SelectImageState
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
  return 'SelectImageState.error(message: $message)';
}


}

/// @nodoc
abstract mixin class _$ErrorCopyWith<$Res> implements $SelectImageStateCopyWith<$Res> {
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

/// Create a copy of SelectImageState
/// with the given fields replaced by the non-null parameter values.
@pragma('vm:prefer-inline') $Res call({Object? message = null,}) {
  return _then(_Error(
null == message ? _self.message : message // ignore: cast_nullable_to_non_nullable
as String,
  ));
}


}

/// @nodoc


class _PreviewDocument implements SelectImageState {
  const _PreviewDocument(this.documentPath);
  

 final  String documentPath;

/// Create a copy of SelectImageState
/// with the given fields replaced by the non-null parameter values.
@JsonKey(includeFromJson: false, includeToJson: false)
@pragma('vm:prefer-inline')
_$PreviewDocumentCopyWith<_PreviewDocument> get copyWith => __$PreviewDocumentCopyWithImpl<_PreviewDocument>(this, _$identity);



@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is _PreviewDocument&&(identical(other.documentPath, documentPath) || other.documentPath == documentPath));
}


@override
int get hashCode => Object.hash(runtimeType,documentPath);

@override
String toString() {
  return 'SelectImageState.previewImage(documentPath: $documentPath)';
}


}

/// @nodoc
abstract mixin class _$PreviewDocumentCopyWith<$Res> implements $SelectImageStateCopyWith<$Res> {
  factory _$PreviewDocumentCopyWith(_PreviewDocument value, $Res Function(_PreviewDocument) _then) = __$PreviewDocumentCopyWithImpl;
@useResult
$Res call({
 String documentPath
});




}
/// @nodoc
class __$PreviewDocumentCopyWithImpl<$Res>
    implements _$PreviewDocumentCopyWith<$Res> {
  __$PreviewDocumentCopyWithImpl(this._self, this._then);

  final _PreviewDocument _self;
  final $Res Function(_PreviewDocument) _then;

/// Create a copy of SelectImageState
/// with the given fields replaced by the non-null parameter values.
@pragma('vm:prefer-inline') $Res call({Object? documentPath = null,}) {
  return _then(_PreviewDocument(
null == documentPath ? _self.documentPath : documentPath // ignore: cast_nullable_to_non_nullable
as String,
  ));
}


}

// dart format on
