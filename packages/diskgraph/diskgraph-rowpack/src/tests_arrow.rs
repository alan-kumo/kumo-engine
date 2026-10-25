// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::*;
use anyhow::Result;
use arrow_array::{BooleanArray, Float32Array, Int64Array, StringArray};

#[test]
fn test_decode_batch_simple() -> Result<()> {
    // 1. Setup Schema
    let cols = vec![
        RowPackColumn {
            name: "id".into(),
            logical_type: IngestedType::Int64,
            fixed_len: None,
            precision: None,
            scale: None,
        },
        RowPackColumn {
            name: "score".into(),
            logical_type: IngestedType::Float32,
            fixed_len: None,
            precision: None,
            scale: None,
        },
        RowPackColumn {
            name: "label".into(),
            logical_type: IngestedType::String,
            fixed_len: None,
            precision: None,
            scale: None,
        },
        RowPackColumn {
            name: "active".into(),
            logical_type: IngestedType::Bool,
            fixed_len: None,
            precision: None,
            scale: None,
        },
    ];

    let mut builder = RowPackContextBuilder::new(cols, RowPackConfig::default());

    // 2. Observe some data to train context
    let rows = vec![
        vec![
            ValueRef::Int64(1),
            ValueRef::Float32(0.5),
            ValueRef::String("A"),
            ValueRef::Bool(true),
        ],
        vec![
            ValueRef::Int64(2),
            ValueRef::Float32(1.5),
            ValueRef::String("B"),
            ValueRef::Bool(false),
        ],
        vec![
            ValueRef::Int64(3),
            ValueRef::Null,
            ValueRef::String("C"),
            ValueRef::Bool(true),
        ], // Null float
    ];

    for r in &rows {
        builder.observe_row(r)?;
    }

    let ctx = builder.finish()?;
    let encoder = RowPackEncoder::new(ctx.clone())?;
    let mut scratch = RowPackScratch::new(&ctx);

    // 3. Encode Rows
    let mut encoded_rows = Vec::new();
    for r in &rows {
        let mut out = Vec::new();
        encoder.encode_row_into(r, &mut out, &mut scratch)?;
        encoded_rows.push(out);
    }

    let encoded_slices: Vec<&[u8]> = encoded_rows.iter().map(|v| v.as_slice()).collect();

    // 4. Decode Batch
    let decoder = RowPackDecoder::new(ctx)?;

    // Decode all columns
    let batch = decoder.decode_batch_to_arrow(&encoded_slices, &[0, 1, 2, 3])?;

    assert_eq!(batch.num_rows(), 3);
    assert_eq!(batch.num_columns(), 4);

    // 5. Verify Data
    let ids = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(ids.value(0), 1);
    assert_eq!(ids.value(1), 2);
    assert_eq!(ids.value(2), 3);

    let scores = batch
        .column(1)
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap();
    assert_eq!(scores.value(0), 0.5);
    assert_eq!(scores.value(1), 1.5);
    assert!(scores.is_null(2));

    let labels = batch
        .column(2)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(labels.value(0), "A");
    assert_eq!(labels.value(1), "B");
    assert_eq!(labels.value(2), "C");

    let active = batch
        .column(3)
        .as_any()
        .downcast_ref::<BooleanArray>()
        .unwrap();
    assert!(active.value(0));
    assert!(!active.value(1));
    assert!(active.value(2));

    Ok(())
}

#[test]
fn test_decode_batch_subset() -> Result<()> {
    // Setup same data...
    let cols = vec![
        RowPackColumn {
            name: "id".into(),
            logical_type: IngestedType::Int64,
            fixed_len: None,
            precision: None,
            scale: None,
        },
        RowPackColumn {
            name: "score".into(),
            logical_type: IngestedType::Float32,
            fixed_len: None,
            precision: None,
            scale: None,
        },
    ];
    let mut builder = RowPackContextBuilder::new(cols, RowPackConfig::default());
    let rows = vec![
        vec![ValueRef::Int64(10), ValueRef::Float32(10.0)],
        vec![ValueRef::Int64(20), ValueRef::Float32(20.0)],
    ];
    for r in &rows {
        builder.observe_row(r)?;
    }
    let ctx = builder.finish()?;
    let encoder = RowPackEncoder::new(ctx.clone())?;
    let mut scratch = RowPackScratch::new(&ctx);
    let mut encoded_rows = Vec::new();
    for r in &rows {
        let mut out = Vec::new();
        encoder.encode_row_into(r, &mut out, &mut scratch)?;
        encoded_rows.push(out);
    }
    let encoded_slices: Vec<&[u8]> = encoded_rows.iter().map(|v| v.as_slice()).collect();
    let decoder = RowPackDecoder::new(ctx)?;

    // Decode only "score" (column index 1)
    let batch = decoder.decode_batch_to_arrow(&encoded_slices, &[1])?;
    assert_eq!(batch.num_columns(), 1);
    let scores = batch
        .column(0)
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap();
    assert_eq!(scores.value(0), 10.0);
    assert_eq!(scores.value(1), 20.0);
    assert_eq!(batch.schema().field(0).name(), "score");

    Ok(())
}

#[test]
fn timestamp_nanoseconds_roundtrip_preserves_boundaries_and_nulls() -> Result<()> {
    use arrow_array::{Array, TimestampNanosecondArray};
    let columns = vec![RowPackColumn {
        name: "at".into(),
        logical_type: IngestedType::Timestamp,
        fixed_len: None,
        precision: None,
        scale: None,
    }];
    let config = RowPackConfig {
        timestamp_nanoseconds: Some(true),
        ..Default::default()
    };
    let mut builder = RowPackContextBuilder::new(columns, config);
    let values = [-100_000_001, 100_000_001, 900_000_001];
    for value in values {
        builder.observe_row(&[ValueRef::TimestampNanosecond(value)])?;
    }
    let ctx = RowPackContext::from_bytes(&builder.finish()?.to_bytes()?)?;
    assert!(matches!(
        ctx.columns[0].encoding,
        ColumnEncoding::TimestampNanosecond
    ));
    let encoder = RowPackEncoder::new(ctx.clone())?;
    let mut scratch = RowPackScratch::new(&ctx);
    let mut rows = Vec::new();
    for value in values
        .into_iter()
        .map(ValueRef::TimestampNanosecond)
        .chain([ValueRef::Null])
    {
        let mut row = Vec::new();
        encoder.encode_row_into(&[value], &mut row, &mut scratch)?;
        rows.push(row);
    }
    let decoder = RowPackDecoder::new(ctx)?;
    let batch = decoder.decode_batch_to_arrow(&rows, &[0])?;
    let actual = batch
        .column(0)
        .as_any()
        .downcast_ref::<TimestampNanosecondArray>()
        .unwrap();
    assert_eq!(&actual.values()[..3], &values);
    assert!(actual.is_null(3));
    Ok(())
}

#[test]
fn timestamp_nanosecond_access_rejects_overflow_in_both_writers() -> Result<()> {
    use arrow_array::TimestampSecondArray;
    use std::sync::Arc;
    let config = RowPackConfig {
        timestamp_nanoseconds: Some(true),
        ..Default::default()
    };
    let ctx = RowPackContextBuilder::new(
        vec![RowPackColumn {
            name: "at".into(),
            logical_type: IngestedType::Timestamp,
            fixed_len: None,
            precision: None,
            scale: None,
        }],
        config,
    )
    .finish()?;
    let encoder = RowPackEncoder::new(ctx.clone())?;
    let mut access = RowAccess::new(vec![RowAccessHints::default()], RowAccessStats::default());
    access.set_cols(&[Arc::new(TimestampSecondArray::from(vec![Some(i64::MAX)]))]);
    let mut out = Vec::new();
    let mut scratch = RowPackScratch::new(&ctx);
    assert!(encoder
        .encode_row_from_access(&mut access, &mut out, &mut scratch)
        .unwrap_err()
        .to_string()
        .contains("nanosecond range"));
    let (mut tags, mut bools, mut data, mut floats) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    assert!(encoder
        .encode_row_from_access_into_buffers(
            &mut access,
            &mut tags,
            &mut bools,
            &mut data,
            &mut floats,
            &[]
        )
        .unwrap_err()
        .to_string()
        .contains("nanosecond range"));
    Ok(())
}
