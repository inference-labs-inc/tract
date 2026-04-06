use crate::internal::*;
use crate::prelude::DatumType;
use ndarray::*;

#[derive(Debug, Clone, new, Hash, PartialEq, Eq)]
pub struct ScatterNd;

impl Op for ScatterNd {
    fn name(&self) -> StaticName {
        "ScatterNd".into()
    }

    op_as_typed_op!();
}

impl ScatterNd {
    unsafe fn eval_t<T: Datum>(
        &self,
        data: TValue,
        indices: &ArrayViewD<i64>,
        updates: TValue,
    ) -> TractResult<TValue> {
        let mut data = unsafe { data.into_tensor().into_array_unchecked::<T>() };
        let updates_plain = updates.try_as_plain()?;
        let updates_view = unsafe { updates_plain.to_array_view_unchecked::<T>() };
        for coords in tract_ndarray::indices(&indices.shape()[..indices.ndim() - 1]) {
            let mut indices_into_data = indices.view();
            let mut updates = updates_view.view();
            for x in coords.slice() {
                indices_into_data.index_axis_inplace(Axis(0), *x);
                updates.index_axis_inplace(Axis(0), *x);
            }
            let mut data = data.view_mut();
            for x in indices_into_data {
                data.index_axis_inplace(Axis(0), *x as usize);
            }

            data.assign(&updates)
        }
        let mut tensor = data.into_tensor();
        unsafe { tensor.set_datum_type(updates.datum_type()) };
        Ok(tensor.into_tvalue())
    }
}

impl TypedOp for ScatterNd {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(inputs[0].datum_type.fact(inputs[0].shape.to_tvec())))
    }
}

impl EvalOp for ScatterNd {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (data, indices, updates) = args_3!(inputs);
        let indices = indices.cast_to::<i64>()?;
        let indices = indices.to_plain_array_view::<i64>()?;
        let (data, updates) =
            if data.datum_type() == DatumType::TDim || updates.datum_type() == DatumType::TDim {
                let data = if data.datum_type() == DatumType::TDim {
                    data.cast_to::<i64>()?.into_owned().into_tvalue()
                } else {
                    data
                };
                let updates = if updates.datum_type() == DatumType::TDim {
                    updates.cast_to::<i64>()?.into_owned().into_tvalue()
                } else {
                    updates
                };
                (data, updates)
            } else {
                (data, updates)
            };
        if data.datum_type() != updates.datum_type() {
            bail!(
                "Data and update must be of the same type, got {:?} and {:?}",
                data.datum_type(),
                updates.datum_type()
            );
        }
        unsafe {
            Ok(tvec!(dispatch_datum_by_size!(Self::eval_t(data.datum_type())(
                self, data, &indices, updates
            ))?))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scatter_nd_i64() {
        let data = tensor2(&[[1i64, 2, 3, 4], [5, 6, 7, 8]]);
        let indices = tensor2(&[[0i64], [1]]);
        let updates = tensor2(&[[9i64, 10, 11, 12], [13, 14, 15, 16]]);
        let result = ScatterNd.eval(tvec!(data.into(), indices.into(), updates.into())).unwrap();
        assert_eq!(*result[0], tensor2(&[[9i64, 10, 11, 12], [13, 14, 15, 16]]));
    }

    #[test]
    fn scatter_nd_tdim_data_and_updates() {
        let data = tensor2(&[[1i64, 2], [3, 4]]).cast_to_dt(DatumType::TDim).unwrap().into_owned();
        let indices = tensor2(&[[0i64, 1]]);
        let updates = tensor1(&[99i64]).cast_to_dt(DatumType::TDim).unwrap().into_owned();
        let result = ScatterNd.eval(tvec!(data.into(), indices.into(), updates.into())).unwrap();
        assert_eq!(result[0].datum_type(), DatumType::I64);
        assert_eq!(result[0].try_as_plain().unwrap().as_slice::<i64>().unwrap(), &[1, 99, 3, 4]);
    }

    #[test]
    fn scatter_nd_tdim_updates_i64_data() {
        let data = tensor2(&[[0i64, 0], [0, 0]]);
        let indices = tensor2(&[[1i64, 0]]);
        let updates = tensor1(&[42i64]).cast_to_dt(DatumType::TDim).unwrap().into_owned();
        let result = ScatterNd.eval(tvec!(data.into(), indices.into(), updates.into())).unwrap();
        assert_eq!(result[0].datum_type(), DatumType::I64);
        assert_eq!(result[0].try_as_plain().unwrap().as_slice::<i64>().unwrap(), &[0, 0, 42, 0]);
    }

    #[test]
    fn scatter_nd_i64_updates_tdim_data() {
        let data = tensor2(&[[1i64, 2], [3, 4]]).cast_to_dt(DatumType::TDim).unwrap().into_owned();
        let indices = tensor2(&[[0i64, 0]]);
        let updates = tensor1(&[77i64]);
        let result = ScatterNd.eval(tvec!(data.into(), indices.into(), updates.into())).unwrap();
        assert_eq!(result[0].datum_type(), DatumType::I64);
        assert_eq!(result[0].try_as_plain().unwrap().as_slice::<i64>().unwrap(), &[77, 2, 3, 4]);
    }
}
