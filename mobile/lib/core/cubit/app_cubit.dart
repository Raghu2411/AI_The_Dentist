import 'dart:async';

import 'package:flutter_bloc/flutter_bloc.dart';

abstract class AppCubit<State> extends Cubit<State> {
  AppCubit(super.initialState);

  @override
  void emit(State state) {
    if (isClosed) return;
    super.emit(state);
  }
}
